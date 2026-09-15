"""CubeEngine: evaluate expressions over a CubeView to ``(D, S)`` panels.

Rounds come from ``schedule.plan_rounds``; each cube round streams the view
in its orientation, evaluates the round's nodes chunk by chunk with CSE and
refcount-based freeing, applies the stock-day tradability mask to every
computed node (a NaN suspended day must never come back as a finite value
through ``is_gt0`` or a NaN-aware reducer), and hands results on:

- roots and degenerate nodes crossing modes -> in-memory ``(S, D)`` panels
  (column 0; exact for minute-degenerate values);
- full-width nodes crossing modes -> a ``(S, D, M)`` float32 memmap in
  ``workdir`` (deleted at the end).

Panel rounds run whole ``(D, S)`` tensors on ``device``.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..axes import MODE_D, MODE_P, MODE_S
from ..expression import Constant, Expression, Feature
from ..store import CubeView
from .dag import Dag, Node
from .kernels import apply_cube, apply_panel
from .schedule import Plan, Round, plan_rounds

ProgressFn = Callable[[int, int, int, int], None]   # (round_idx, n_rounds, done_units, total_units)


@dataclass
class EvalResult:
    panels: List[Optional[np.ndarray]]        # per root: (D, S) float32 or None
    errors: Dict[int, str]                    # root index -> message
    plan: Plan
    seconds: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)


class CubeEngine:
    def __init__(self, view: CubeView, device: str = "cpu", chunk_symbols: Optional[int] = None,
                 chunk_days: Optional[int] = None, chunk_bytes: int = 2 << 30,
                 apply_mask: bool = True, mask_policy: str = "boundaries",
                 workdir: Optional[os.PathLike] = None, torch_threads: Optional[int] = None) -> None:
        """``mask_policy``: "all" re-applies the stock-day mask at every computed
        node; "boundaries" (default) only where a fake finite value could leak
        -- inputs of day-axis / NT operators and values leaving a round -- which
        is provably the same result (an intraday node only affects its own
        stock-day) at a fraction of the masking cost. Locked by
        tests/test_engine.py::test_mask_policies_agree."""
        if mask_policy not in ("all", "boundaries"):
            raise ValueError("mask_policy must be 'all' or 'boundaries'")
        self.view = view
        self.device = torch.device(device)
        self.chunk_symbols = chunk_symbols
        self.chunk_days = chunk_days
        self.chunk_bytes = int(chunk_bytes)
        self.apply_mask = bool(apply_mask)
        self.mask_policy = mask_policy
        self.workdir = None if workdir is None else Path(workdir)
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))
        self._tradable = torch.from_numpy(view.tradable()).to(self.device)   # (S, D) bool
        self._mask_needed = self.apply_mask and not bool(view.tradable().all())

    # ---- planning --------------------------------------------------------------
    def plan(self, exprs: Sequence[Expression]) -> Plan:
        return plan_rounds(Dag(exprs))

    def _chunk_sizes(self, dag: Dag):
        S, D, M = self.view.n_symbols, self.view.n_days, self.view.n_minutes
        wmax = max((n.expr.window or 0 for n in dag.computed if not (n.spec and n.spec.restores_m)), default=0)
        overhead = 12 + min(wmax, 64)
        per_symbol = D * M * 4 * overhead
        per_day = S * M * 4 * overhead
        cs = self.chunk_symbols or max(1, min(S, self.chunk_bytes // max(per_symbol, 1)))
        cd = self.chunk_days or max(1, min(D, self.chunk_bytes // max(per_day, 1)))
        return int(min(cs, S)), int(min(cd, D))

    # ---- evaluation --------------------------------------------------------------
    def evaluate(self, exprs: Sequence[Expression], on_progress: Optional[ProgressFn] = None,
                 tolerant: bool = False) -> EvalResult:
        t0 = time.perf_counter()
        exprs = list(exprs)
        errors: Dict[int, str] = {}
        for i, e in enumerate(exprs):
            if not e.shape.degenerate:
                msg = f"root is not one value per stock-day (minute axis still live): {e!r}"
                if not tolerant:
                    raise ValueError(msg)
                errors[i] = msg
        live = [e for i, e in enumerate(exprs) if i not in errors]
        live_index = [i for i in range(len(exprs)) if i not in errors]
        dag = Dag(live)
        plan = plan_rounds(dag)
        S, D = self.view.n_symbols, self.view.n_days
        panels: Dict[str, np.ndarray] = {}          # key -> (S, D) float32
        bounds: Dict[str, np.memmap] = {}           # key -> (S, D, M) memmap
        tmp = None
        if plan.boundaries:
            tmp = tempfile.TemporaryDirectory(dir=self.workdir, prefix="alphacube_")
        stats = {"chunks": 0, "boundary_bytes": 0.0, "rounds": len(plan.rounds)}
        try:
            cs, cd = self._chunk_sizes(dag)
            total_units = sum((S if r.mode == MODE_S else D if r.mode == MODE_D else 1) for r in plan.rounds)
            done_units = 0
            for r_idx, rnd in enumerate(plan.rounds):
                if rnd.mode == MODE_P:
                    self._run_panel_round(rnd, panels, tolerant)
                    done_units += 1
                else:
                    step = cs if rnd.mode == MODE_S else cd
                    n = S if rnd.mode == MODE_S else D
                    for lo in range(0, n, step):
                        hi = min(lo + step, n)
                        self._run_cube_chunk(rnd, lo, hi, panels, bounds, tmp, tolerant)
                        stats["chunks"] += 1
                        done_units += hi - lo
                        if on_progress is not None:
                            on_progress(r_idx, len(plan.rounds), done_units, total_units)
                if on_progress is not None:
                    on_progress(r_idx, len(plan.rounds), done_units, total_units)
            out: List[Optional[np.ndarray]] = [None] * len(exprs)
            for node in dag.roots:
                for k in node.root_indices:
                    i = live_index[k]
                    if node.failed:
                        errors[i] = node.error or "evaluation failed"
                        continue
                    if node.is_leaf:
                        p = self._leaf_panel(node.expr)
                    else:
                        p = panels[node.key]
                    out[i] = np.ascontiguousarray(p.T, dtype=np.float32)     # (D, S)
            for b in bounds.values():
                stats["boundary_bytes"] += float(b.nbytes)
        finally:
            for b in list(bounds.values()):
                del b
            bounds.clear()
            if tmp is not None:
                tmp.cleanup()
        return EvalResult(panels=out, errors=errors, plan=plan, seconds=time.perf_counter() - t0, stats=stats)

    # ---- helpers -----------------------------------------------------------------
    def _leaf_panel(self, expr: Expression) -> np.ndarray:
        """A bare degenerate leaf root (a Constant) as an (S, D) panel."""
        S, D = self.view.n_symbols, self.view.n_days
        if isinstance(expr, Constant):
            p = np.full((S, D), expr.value, dtype=np.float32)
            if self._mask_needed:
                p[~self.view.tradable()] = np.nan
            return p
        raise ValueError(f"leaf {expr!r} is not a daily value")

    def _mask_cube(self, t, tradable: torch.Tensor):
        if isinstance(t, list):
            return [self._mask_cube(x, tradable) for x in t]
        if not self._mask_needed or t.shape[0] == 1 and t.shape[1] == 1:
            return t
        if t.dtype == torch.bool:
            return t & tradable.unsqueeze(-1)
        return torch.where(tradable.unsqueeze(-1), t, torch.full((), float("nan"), dtype=t.dtype, device=t.device))

    def _load_leaf_cube(self, expr: Expression, mode: str, lo: int, hi: int, tradable: torch.Tensor) -> torch.Tensor:
        if isinstance(expr, Constant):
            return torch.full((1, 1, 1), expr.value, dtype=torch.float32, device=self.device)
        arr = self.view.read_s(expr.field, lo, hi) if mode == MODE_S else self.view.read_d(expr.field, lo, hi)
        t = torch.from_numpy(arr).to(self.device)
        return self._mask_cube(t, tradable)

    def _panel_slice(self, key: str, panels: Dict[str, np.ndarray], mode: str, lo: int, hi: int) -> torch.Tensor:
        p = panels[key]
        sl = p[lo:hi, :] if mode == MODE_S else p[:, lo:hi]
        return torch.from_numpy(np.ascontiguousarray(sl)).to(self.device).unsqueeze(-1)

    def _bound_slice(self, key: str, bounds: Dict[str, np.memmap], mode: str, lo: int, hi: int) -> torch.Tensor:
        b = bounds[key]
        sl = b[lo:hi] if mode == MODE_S else b[:, lo:hi]
        return torch.from_numpy(np.ascontiguousarray(sl)).to(self.device)

    def _run_cube_chunk(self, rnd: Round, lo: int, hi: int, panels, bounds, tmp, tolerant: bool) -> None:
        mode = rnd.mode
        S, D, M = self.view.n_symbols, self.view.n_days, self.view.n_minutes
        tradable = self._tradable[lo:hi, :] if mode == MODE_S else self._tradable[:, lo:hi]
        values: Dict[str, torch.Tensor] = {}
        in_round = {n.key for n in rnd.nodes}
        remaining_uses: Dict[str, int] = {}
        for n in rnd.nodes:
            for c in n.children:
                if not c.is_leaf and c.key in in_round:
                    remaining_uses[c.key] = remaining_uses.get(c.key, 0) + 1

        def operand(c: Node) -> torch.Tensor:
            if c.is_leaf:
                return self._load_leaf_cube(c.expr, mode, lo, hi, tradable)
            if c.key in values:
                return values[c.key]
            if c.key in panels:                       # degenerate hand-off from another round
                return self._panel_slice(c.key, panels, mode, lo, hi)
            if c.key in bounds:                       # full-width boundary from another round
                return self._bound_slice(c.key, bounds, mode, lo, hi)
            raise RuntimeError(f"operand {c.key} unavailable in round {rnd}")

        for n in rnd.nodes:
            if n.failed or any(c.failed for c in n.children):
                n.failed = True
                n.error = n.error or next((c.error for c in n.children if c.failed), "dependency failed")
                continue
            try:
                vals = [operand(c) for c in n.children]
                with torch.no_grad():
                    out = apply_cube(n.expr, vals, mode)
                if self.mask_policy == "all" or n.mask_out:
                    out = self._mask_cube(out, tradable)
                if isinstance(out, list) and (n.panel_out or n.boundary or n.is_root):
                    raise TypeError(f"{n.spec.name} returns a list of tensors; only another operator "
                                    "may consume it (it cannot be a factor root or cross a round)")
            except Exception as exc:  # noqa: BLE001
                if not tolerant:
                    raise
                n.failed, n.error = True, f"{n.spec.name}: {type(exc).__name__}: {exc}"
                continue
            values[n.key] = out
            if n.panel_out:
                p = panels.get(n.key)
                if p is None:
                    p = panels[n.key] = np.full((S, D), np.nan, dtype=np.float32)
                col = out[..., 0].float() if out.dtype == torch.bool else out[..., 0]
                if col.shape[0] == 1 and col.shape[1] == 1:
                    col = col.expand(hi - lo if mode == MODE_S else S, D if mode == MODE_S else hi - lo)
                arr = col.detach().cpu().numpy()
                if self._mask_needed:
                    arr = np.where(tradable.cpu().numpy(), arr, np.nan)
                if mode == MODE_S:
                    p[lo:hi, :] = arr
                else:
                    p[:, lo:hi] = arr
            if n.boundary:
                arr = (out.float() if out.dtype == torch.bool else out).detach().cpu().numpy()
                b = bounds.get(n.key)
                if b is None:
                    # width comes from the RESULT: a width-changing call (in_slice) is narrower than M
                    path = Path(tmp.name) / f"bound_{len(bounds):04d}.npy"
                    b = bounds[n.key] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                                   shape=(S, D, int(arr.shape[-1])))
                if mode == MODE_S:
                    b[lo:hi] = arr
                else:
                    b[:, lo:hi] = arr
            for c in n.children:
                if c.key in remaining_uses:
                    remaining_uses[c.key] -= 1
                    if remaining_uses[c.key] == 0 and not c.is_root:
                        values.pop(c.key, None)
        for b in bounds.values():
            b.flush()

    def _run_panel_round(self, rnd: Round, panels: Dict[str, np.ndarray], tolerant: bool) -> None:
        tradable_ds = self._tradable.transpose(0, 1)          # (D, S)
        values: Dict[str, torch.Tensor] = {}

        def operand(c: Node) -> torch.Tensor:
            if isinstance(c.expr, Constant):
                return torch.full((1, 1), c.expr.value, dtype=torch.float32, device=self.device)
            if c.key in values:
                return values[c.key]
            if c.key in panels:
                return torch.from_numpy(np.ascontiguousarray(panels[c.key].T)).to(self.device)
            raise RuntimeError(f"panel operand {c.key} unavailable")

        for n in rnd.nodes:
            if n.failed or any(c.failed for c in n.children):
                n.failed = True
                n.error = n.error or next((c.error for c in n.children if c.failed), "dependency failed")
                continue
            try:
                vals = [operand(c) for c in n.children]
                with torch.no_grad():
                    out = apply_panel(n.expr, vals)
                if isinstance(out, list):
                    raise TypeError(f"{n.spec.name} returns a list of tensors; not usable on the panel face")
                if self._mask_needed and not (out.shape[0] == 1 and out.shape[1] == 1):
                    if out.dtype == torch.bool:
                        out = out & tradable_ds
                    else:
                        out = torch.where(tradable_ds, out, torch.full((), float("nan"), dtype=out.dtype, device=out.device))
            except Exception as exc:  # noqa: BLE001
                if not tolerant:
                    raise
                n.failed, n.error = True, f"{n.spec.name}: {type(exc).__name__}: {exc}"
                continue
            values[n.key] = out
            if n.is_root or n.panel_out:
                if out.dtype == torch.bool:
                    out = out.float()
                if out.shape[0] == 1 and out.shape[1] == 1:
                    out = out.expand(self.view.n_days, self.view.n_symbols)
                    if self._mask_needed:
                        out = torch.where(tradable_ds, out, torch.full((), float("nan"), dtype=out.dtype, device=out.device))
                panels[n.key] = np.ascontiguousarray(out.detach().cpu().numpy().T, dtype=np.float32)


__all__ = ["CubeEngine", "EvalResult"]
