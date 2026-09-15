"""Present the right two-axis view of a cube chunk to each kernel family.

Cube chunks are ``(A, B, M)`` where in S-mode ``A`` is the stock chunk and
``B`` the full day axis; in D-mode ``A`` is the full stock axis and ``B``
the day chunk. Collapsed operands are ``(A, B, 1)``; constants ``(1, 1, 1)``.

Family -> view (per-row/per-column math is identical to evaluating one
stock or one day at a time, which is what makes chunking exact):

- M-axis DT kernels (``in_*``, ``day_*``, ``in_expand``): ``(A*B, M)`` view,
  zero-copy; rows are exactly the per-(stock, day) rows.
- NT kernels (``nt_cs_*`` / ``nt_ts_*``, D-mode only): per-day loop, the ND
  ``(T, N)`` kernel called on that day's ``(M, S)`` face. This is the
  definition of the face, and the only form that is chunk-invariant: batching
  days into one call changes float32 reduction order (and is plain wrong for
  the two column-mixing kernels ``ts_pearson``/``ts_spearman``).
- D-axis DT kernels (``cross_*``, S-mode only): per-stock loop on ``(D, M)``
  -- the previous engine measured trailing-ULP differences when day-rolling
  kernels were batched as ``(D, C*M)``, so the loop keeps them bit-exact.
- ND panel kernels: ``(D, S)`` directly.
- elementwise: torch broadcasting on the cube.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..axes import MODE_D, MODE_P, MODE_S, OpSpec
from ..expression import Expression, GeneralCall


def _as_float(t: torch.Tensor) -> torch.Tensor:
    """Every node exit is float32 -- the contract of the old per-node numpy
    round-trip that the legacy corpus was validated against (a boolean or
    integer result fed to a NaN-aware reducer must be a float, else the NaN
    pad overflows). Kernels that want a boolean mask get it back via the
    parameter-name cast in ``_call``."""
    if not t.is_floating_point():
        return t.float()
    return t


_MASK_POSITIONS: dict = {}


def _mask_positions(fn) -> tuple:
    """Positional indices of parameters named like a mask (``valid_mask``, ``mask``)."""
    key = id(fn)
    if key not in _MASK_POSITIONS:
        import inspect
        try:
            names = [q.name for q in inspect.signature(fn).parameters.values()
                     if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            names = []
        _MASK_POSITIONS[key] = tuple(i for i, n in enumerate(names) if n == "mask" or n.endswith("_mask"))
    return _MASK_POSITIONS[key]


def _cast_masks(fn, call_args: list) -> list:
    pos = _mask_positions(fn)
    if not pos:
        return call_args
    out = list(call_args)
    for i in pos:
        if i < len(out) and isinstance(out[i], torch.Tensor) and out[i].dtype != torch.bool:
            out[i] = torch.nan_to_num(out[i], nan=0.0) != 0
    return out


def _expand_like(x: torch.Tensor, ref_shape) -> torch.Tensor:
    """Broadcast ``x`` (constant or collapsed) to ``ref_shape`` as a view."""
    if tuple(x.shape) == tuple(ref_shape):
        return x
    return x.expand(*ref_shape)


def _full_shape(vals: List[torch.Tensor]):
    """Widest shape among operands (broadcast target)."""
    best = None
    for v in vals:
        if best is None or v.numel() > best.numel():
            best = v
    return tuple(best.shape)


def _norm(out):
    """Kernel output -> tensor or list of tensors, dtype-normalised."""
    if isinstance(out, list):
        return [_as_float(t) for t in out]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"kernel returned {type(out).__name__}, not a tensor")
    return _as_float(out)


def _call(expr: Expression, args: List, window):
    """Invoke the kernel with tensor operands re-interleaved with any literal
    arguments (GeneralCall) or the structural window (rolling operators)."""
    spec = expr.spec
    if isinstance(expr, GeneralCall):
        it = iter(args)
        call_args = []
        for a in expr.args:
            if isinstance(a, tuple):
                call_args.append([next(it) for _ in a])
            elif isinstance(a, Expression):
                call_args.append(next(it))
            else:
                call_args.append(a)
        return spec.fn(*_cast_masks(spec.fn, call_args))
    if spec.category in ("rolling", "pair_rolling"):
        return spec.fn(*_cast_masks(spec.fn, list(args)), int(window))
    return spec.fn(*_cast_masks(spec.fn, list(args)))


def _tensors(vals: List) -> List[torch.Tensor]:
    return [v for v in vals if isinstance(v, torch.Tensor)]


def _align_widths(vals: List, skip: frozenset = frozenset()) -> List:
    """Broadcast width-1 (day-collapsed / constant) operands to the common
    minute width; two genuinely different widths is a real mistake in the
    expression (mismatched in_slice ranges) and is reported as one."""
    counted = [v for i, v in enumerate(vals) if i not in skip and isinstance(v, torch.Tensor)]
    if not counted:
        return vals
    widths = {v.shape[-1] for v in counted}
    m = max(widths)
    bad = sorted(w for w in widths if w not in (1, m))
    if bad:
        raise ValueError(f"cannot combine operands of minute-axis widths {sorted(widths)}: only a width-1 "
                         "operand can broadcast against a wider one")
    lead = max((tuple(v.shape[:-1]) for v in counted), key=lambda t: t[0] * t[1])
    out = []
    for i, v in enumerate(vals):
        if not isinstance(v, torch.Tensor) or i in skip:
            out.append(v)
        else:
            target = lead + (v.shape[-1] if v.shape[-1] == m else m,)
            out.append(v if tuple(v.shape) == target else v.expand(*target).contiguous())
    return out


def _pre_collapse(spec: OpSpec, vals: List) -> List:
    """in_expand on a constant-widened degenerate panel: use column 0 (every column is equal)."""
    if spec.restores_m:
        return [v[..., :1].contiguous() if isinstance(v, torch.Tensor) and v.shape[-1] != 1 else v for v in vals]
    return vals


def _per_row_view(vals: List, A: int, B: int) -> List:
    out = []
    for v in vals:
        if isinstance(v, torch.Tensor):
            out.append(v.reshape(A * B, v.shape[-1]))
        elif isinstance(v, list):
            out.append([t.reshape(A * B, t.shape[-1]) for t in v])
        else:
            out.append(v)
    return out


def _from_rows(out, A: int, B: int):
    if isinstance(out, list):
        return [t.reshape(A, B, -1) for t in out]
    return out.reshape(A, B, -1)


def apply_cube(expr: Expression, vals: List, mode: str):
    """Apply ``expr``'s operator to cube-mode operands (A, B, M|1) -> (A, B, M|1).
    Operands may be tensors or lists of tensors (multi-regression intermediates)."""
    spec = expr.spec
    window = expr.window
    if any(isinstance(v, list) for v in vals) and spec.face != "nt" and spec.compute_axis != "D":
        return _norm(_call(expr, vals, window))         # list consumers are positional/elementwise
    if spec.family == "elementwise":
        return _norm(_call(expr, vals, window))
    exempt = getattr(expr, "exempt", frozenset())
    vals = _pre_collapse(spec, vals)
    tens = _tensors(vals)
    if not tens:
        return _norm(_call(expr, vals, window))
    ref = _full_shape(tens)
    A, B = ref[0], ref[1]
    if spec.face == "nt":
        if mode != MODE_D:
            raise RuntimeError(f"{spec.name} (NT face) must run in D-mode, got {mode}")
        args = _align_widths(vals, exempt)
        outs = []
        for c in range(B):                                            # one (M, S) face per day
            face = [a[:, c, :].transpose(0, 1).contiguous() if isinstance(a, torch.Tensor) else a for a in args]
            outs.append(_as_float(_call(expr, face, window)).transpose(0, 1))
        return torch.stack(outs, dim=1).contiguous()                  # (S, B, M)
    if spec.compute_axis == "M":
        if spec.restores_m:
            x = tens[0]
            return _norm(_call(expr, [x.reshape(A * B, 1)] + [v for v in vals[1:] if not isinstance(v, torch.Tensor)],
                               window)).reshape(A, B, -1)
        args = _align_widths(vals, exempt)
        return _from_rows(_norm(_call(expr, _per_row_view(args, A, B), window)), A, B)
    if spec.compute_axis == "D":
        if mode != MODE_S:
            raise RuntimeError(f"{spec.name} (D-axis) must run in S-mode, got {mode}")
        args = _align_widths(vals, exempt)
        outs = []
        for a in range(A):
            outs.append(_norm(_call(expr, [x[a] if isinstance(x, torch.Tensor) else
                                           ([t[a] for t in x] if isinstance(x, list) else x) for x in args], window)))
        if isinstance(outs[0], list):
            return [torch.stack([o[k] for o in outs], dim=0) for k in range(len(outs[0]))]
        return torch.stack(outs, dim=0)
    raise RuntimeError(f"cannot apply {spec.name} in cube mode")


def apply_panel(expr: Expression, vals: List):
    """Apply ``expr``'s operator to (D, S) panels (constants are (1, 1))."""
    spec = expr.spec
    if spec.family == "elementwise" or any(isinstance(v, list) for v in vals):
        return _norm(_call(expr, vals, expr.window))
    tens = _tensors(vals)
    ref = _full_shape(tens) if tens else None
    args = [_expand_like(v, ref) if isinstance(v, torch.Tensor) else v for v in vals]
    return _norm(_call(expr, args, expr.window))


__all__ = ["apply_cube", "apply_panel", "MODE_S", "MODE_D", "MODE_P"]
