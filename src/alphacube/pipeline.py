"""One-shot mining run: cube -> label -> pool -> miner -> metrics -> combination
leaderboard -> run directory. Used by ``cli mine``, ``train.py`` and
``scripts/compare_miners.py`` so all three share one code path."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

MINERS = ("gp", "alphagen", "alphaqcm", "alphaforge", "alphasage", "warmstart")
POOLS = ("mse", "meanstd", "dynamic")
LABELS = ("open_vwap15", "open_vwap15_buyable", "close", "volatility", "downside_volatility", "rank", "class", "spike", "crash")
# Columns of the combination tables (block reports and the final one), when present.
# *_smooth: the same combination traded through an EMA whose half-life was picked on
# the training days (post_combine.smoothed_metrics).
_TABLE_COLUMNS = ("ic_mean", "icir", "rank_ic", "sharpe", "sharpe_net", "sharpe_capped", "turnover",
                  "halflife", "sharpe_smooth", "turnover_smooth", "coverage", "error")


@dataclass
class RunResult:
    run_dir: Optional[Path]
    pool_size: int
    added: int
    pool_metrics: Dict[str, float]
    members: List[dict]
    combination: Optional[Any]          # DataFrame or None
    seconds: float
    engine_seconds: float
    rejected: Dict[str, int]
    summary: str


class _Lab:
    def __init__(self, desc, compute, realized) -> None:
        self._desc, self._compute, self._realized = desc, compute, realized

    def describe(self) -> str:
        return self._desc

    def compute(self, view):
        return self._compute(view)

    def realized(self, view):
        return self._realized(view)


def make_label(kind: str, horizon: int, params: Optional[Dict[str, float]] = None) -> _Lab:
    """A label with ``compute(view) -> (D, S)`` and ``realized(view)`` (the P&L leg)."""
    from .label import Label

    if kind in ("open_vwap15", "close"):
        lab = Label(kind, horizon)
        return _Lab(lab.describe(), lab.compute, lab.compute)
    if kind == "open_vwap15_buyable":                  # 剔除次日开盘封涨停、买不进的那部分收益
        lab = Label("open_vwap15", horizon, buyable=True)
        return _Lab(lab.describe(), lab.compute, lab.compute)
    from .daily import daily_from_view
    from .metrics import daily_returns_from_close
    from .models import targets as T

    p = dict(params or {})

    def compute(view):
        r = daily_returns_from_close(daily_from_view(view, fields=["Close"], open_vwap15=False))
        if kind == "volatility":
            return T.forward_volatility(r, horizon, **p)
        if kind == "downside_volatility":
            return T.forward_downside_volatility(r, horizon, **p)
        if kind == "rank":
            return T.forward_return_rank(r, horizon, **p)
        if kind == "class":
            return T.forward_return_class(r, horizon, **p) - 1.0
        return T.forward_event_label(r, horizon, kind=kind, **p)

    return _Lab(f"{kind}(h={horizon}, {p})", compute, lambda v: Label("close", horizon).compute(v))


def ban_exemplars_from(paths: Sequence[Union[str, Path]], keys: Sequence[str]) -> List[Any]:
    """Rank-ban exemplars for structures banned by an earlier run: every expression
    in those checkpoint CSVs (``archive_genNNN.csv``) that contains one of ``keys``,
    best fitness first, each once. Rows that no longer parse are skipped."""
    import csv

    from .mining.generator import one_layer_keys
    from .parser import parse

    wanted = set(keys)
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    e = parse(r["expression"])
                except Exception:  # noqa: BLE001 - an old archive may hold a retired operator
                    continue
                if one_layer_keys(e) & wanted:
                    rows.append((float(r.get("fitness") or 0.0), e))
    seen, out = set(), []
    for _, e in sorted(rows, key=lambda t: -t[0]):
        if e.key() not in seen:
            seen.add(e.key())
            out.append(e)
    return out


def build_pool(kind: str, capacity: int, target, realized, dedup: Optional[float] = 0.99,
               min_coverage: float = 0.5, meanstd_lr: float = 0.05, meanstd_steps: int = 200,
               dynamic_window: int = 60, min_scored_days: float = 0.5):
    from .mining.pool import DynamicRollingPool, MeanStdPool, MsePool

    kw = dict(dedup_threshold=dedup, min_coverage=min_coverage, min_scored_days=min_scored_days,
              realized_returns=realized)
    if kind == "mse":
        return MsePool(capacity, target, **kw)
    if kind == "meanstd":
        return MeanStdPool(capacity, target, lr=meanstd_lr, n_steps=meanstd_steps, **kw)
    if kind == "dynamic":
        return DynamicRollingPool(capacity, target, window=dynamic_window, **kw)
    raise ValueError(f"unknown pool {kind!r}; choose from {POOLS}")


def align_universe(mask: np.ndarray, dates: Sequence[str], symbols: Sequence[str],
                   view_dates: Sequence[str], view_symbols: Sequence[str]) -> np.ndarray:
    """把 (D, S) 股票池掩码按名字对齐到一个视图上;视图里掩码没覆盖的日子/股票一律算池外。"""
    di = {d: i for i, d in enumerate(dates)}
    si = {s: i for i, s in enumerate(symbols)}
    out = np.zeros((len(view_dates), len(view_symbols)), dtype=bool)
    rows = [(r, di[d]) for r, d in enumerate(view_dates) if d in di]
    cols = [(c, si[s]) for c, s in enumerate(view_symbols) if s in si]
    if rows and cols:
        r_v, r_m = map(list, zip(*rows))
        c_v, c_m = map(list, zip(*cols))
        out[np.ix_(r_v, c_v)] = mask[np.ix_(r_m, c_m)]
    return out


class UniverseEngine:
    """引擎外面套一层:每个面板在池外的格子上置 NaN。奖励里的 IC、十分位书的换手、池子的合成
    就都只看池子里的票(只把标签置空不够:换手是按因子值本身排名建书的)。其余属性原样转给引擎。"""

    def __init__(self, engine: Any, member: np.ndarray) -> None:
        self._engine, self._member = engine, member

    def evaluate(self, exprs, *args, **kw):
        res = self._engine.evaluate(exprs, *args, **kw)
        res.panels = [None if p is None else np.where(self._member, p, np.nan).astype(p.dtype)
                      for p in res.panels]
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def load_universe_file(path: Union[str, Path]):
    """(掩码, 日期, 股票):scripts/universe.py 写的 .npy 和旁边的 .json(日期、股票顺序)。"""
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    mask = np.load(p)
    if mask.shape != (len(meta["dates"]), len(meta["symbols"])):
        raise ValueError(f"{p}: mask {mask.shape} vs meta {len(meta['dates'])} x {len(meta['symbols'])}")
    return mask, meta["dates"], meta["symbols"]


def run_mining(
    store: Union[str, Path, Any],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_symbols: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
    warmup: int = 60,
    label: str = "open_vwap15",
    label_params: Optional[Dict[str, float]] = None,
    horizon: int = 1,
    miner: str = "gp",
    budget: int = 480,
    seed: int = 0,
    device: str = "cpu",
    chunk_symbols: Optional[int] = None,
    chunk_days: Optional[int] = None,
    pool: str = "mse",
    pool_capacity: int = 10,
    dedup: Optional[float] = 0.99,
    min_coverage: float = 0.5,
    min_scored_days: float = 0.5,
    cost_per_side: float = 0.0,
    turnover_cap: Optional[float] = None,
    meanstd_lr: float = 0.05,
    meanstd_steps: int = 200,
    dynamic_window: int = 60,
    seed_library: Optional[Union[str, Path]] = None,
    presqueeze: Optional[dict] = None,
    allow_boolean_ops: bool = False,
    allow_resource_risk: bool = False,
    skip_kernels: Sequence[str] = (),
    checkpoint_every: int = 0,
    rolling_block: int = 0,
    ban_top_n: int = 0,
    block_combine: bool = True,
    block_combine_top: int = 120,
    engine_progress_every: float = 0.0,
    workdir: Optional[Union[str, Path]] = None,
    gp_params: Optional[dict] = None,
    rl_params: Optional[dict] = None,
    max_expr_len: int = 20,
    combine_methods: Sequence[str] = ("equal", "ic", "icir", "lstsq", "ridge", "gbdt"),
    combine_train_frac: float = 0.7,
    smooth_halflives: Sequence[float] = (),
    mirror_root: Optional[Union[str, Path]] = None,
    daily_root: Optional[Union[str, Path]] = None,
    universe: Optional[Union[str, Path]] = None,
    out_root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    save_values: bool = True,
    log: Optional[Callable[[str], None]] = print,
) -> RunResult:
    from . import dailyfields
    from .catalog import default_registry
    from .engine.engine import CubeEngine
    from .metrics import backtest_metrics, cs_pearson, ic_summary
    from .mining import (AlphaForgeMiner, AlphaGenMiner, AlphaQCMMiner, AlphaSAGEMiner, BatchRewarder, GPMiner,
                         PPOWarmStartQCMMiner, Vocabulary)
    from .research import warmed_view
    from .store import CubeStore, SegmentedCubeStore, open_cube_store

    say = log or (lambda s: None)
    t0 = time.time()
    if not isinstance(store, (CubeStore, SegmentedCubeStore)):
        store = open_cube_store(store, mirror_root=mirror_root, daily_root=daily_root)
    if store.daily is not None:
        say(f"daily fields {store.daily.root}: {', '.join(dailyfields.available(store.daily))}")
    else:
        say("daily fields: none attached")
    member = None
    if universe is not None:
        if symbols is not None or max_symbols is not None:
            raise ValueError("universe 和 symbols / max_symbols 只能给一个")
        u_mask, u_dates, u_syms = load_universe_file(universe)
        # 引擎只在"这段日子里进过池子"的票上算,截面运算就只在这批票之间做;每天的池外格子另外置空。
        # 只看 [start, end] 里的成分:挖训练段时不该知道之后才进池子的票
        rows = [i for i, d in enumerate(u_dates) if (start is None or d >= start) and (end is None or d <= end)]
        symbols = [s for s, keep in zip(u_syms, u_mask[rows].any(0)) if keep]
    view, trim = warmed_view(store, start, end, warmup=warmup, symbols=symbols, max_symbols=max_symbols)
    lab = make_label(label, horizon, label_params)
    target, realized = lab.compute(view), lab.realized(view)
    if universe is not None:
        member = align_universe(u_mask, u_dates, u_syms, view.dates, view.symbols)
        target = np.where(member, target, np.nan)
        realized = np.where(member, realized, np.nan)
        say(f"universe {universe}: {view.n_symbols} symbols ever in, {member.sum(1).mean():.0f} a day on average")
    say(f"store {getattr(store, 'directory', store)}: {view.dates[0]}..{view.dates[-1]} {view.n_days} days x "
        f"{view.n_symbols} symbols (warm-up {trim}); label {lab.describe()}; tradable {view.tradable().mean():.1%}")
    reg = default_registry()
    pool_obj = build_pool(pool, pool_capacity, target, realized, dedup, min_coverage, meanstd_lr, meanstd_steps,
                          dynamic_window, min_scored_days=min_scored_days)
    engine = CubeEngine(view, device=device, chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                        workdir=workdir)
    if member is not None:
        engine = UniverseEngine(engine, member)
    if seed_library:
        from .legacy import load_seed_library, seed_pool
        exprs, errs = load_seed_library(seed_library, n_minutes=view.n_minutes)
        exprs = [e for _, e in exprs if e.shape.degenerate]
        rewards = seed_pool(pool_obj, exprs, engine)
        say(f"seeded pool with {len(rewards)} of {len(exprs)} library factors ({len(errs)} failed to parse)")
    if presqueeze is not None:
        from .mining.boosting import PresqueezeConfig, feature_boosting_presqueeze, seed_pool_from_presqueeze
        res = feature_boosting_presqueeze(reg, engine, target,
                                          PresqueezeConfig(allow_boolean_ops=allow_boolean_ops,
                                                          allow_resource_risk=allow_resource_risk, **presqueeze))
        rewards = seed_pool_from_presqueeze(pool_obj, res, engine)
        say(f"pre-squeeze: {res.candidates_generated_total} candidates, {res.candidates_evaluated_total} scored, "
            f"{len(res.survivors)} survivors, {len(rewards)} entered the pool")

    _last_tick = [0.0]

    def engine_progress(d):
        """Inside-a-generation progress. Throttled: the engine calls back per chunk,
        which on the production cube is hundreds of times per round."""
        now = time.time()
        if now - _last_tick[0] < engine_progress_every:
            return
        _last_tick[0] = now
        eta = d["eta"]
        eta_s = "?" if eta != eta else f"{eta / 60:.0f}m"
        # "units", not "chunks": the engine counts a round's progress in symbols
        # (S-mode) or days (D-mode), so the total is sum(S or D over rounds) -- e.g.
        # 2 S-rounds + 1 D-round + 3 panel rounds on the production cube is
        # 2*5219 + 1942 + 3 = 12383, which is not a chunk count and reads very wrong
        # as one.
        say(f"      ... {d['trees']} trees ({d['cached']} cached)  round {d['round']}/{d['rounds']}"
            f"  {d['done']}/{d['total']} units  {100 * d['frac']:.0f}%"
            f"  elapsed {d['elapsed'] / 60:.0f}m  eta {eta_s}")

    def progress(n, b, s):
        extra = "  ".join(f"{k} {v:+.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items())
        say(f"  {n:5d}/{b}  {extra}")

    block_reports: List[dict] = []
    miner_ref: List[Any] = []          # set before mine() so on_block can reach the panel cache

    def score_archive(entries, label: str) -> Optional[dict]:
        """Pool + out-of-sample combination over a set of archived expressions.

        Panels come back through the miner's own cache where they are still
        resident, so scoring a block costs little beyond what the search already
        paid; the cumulative view reaches further back and may re-evaluate."""
        from .combine.post_combine import compare, leaderboard

        # block_combine_top 0: every archived expression, not only the leaders
        exprs = [e for e, _ in (entries[:block_combine_top] if block_combine_top else entries)]
        if not exprs or not miner_ref:
            return None
        panels = miner_ref[0]._panels(exprs)
        sub = build_pool(pool, pool_capacity, target, realized, dedup, min_coverage, meanstd_lr,
                         meanstd_steps, dynamic_window, min_scored_days=min_scored_days)
        sub.admit(zip(exprs, panels))                            # gate rejections are expected
        if sub.size == 0:
            say(f"  [{label}] 池为空,跳过")
            return None
        res, signs = compare(sub.z_columns(), target, realized, methods=tuple(combine_methods),
                             train_frac=combine_train_frac, align=True, seed=seed,
                             cost_per_side=cost_per_side, turnover_cap=turnover_cap,
                             smooth_halflives=smooth_halflives)
        tbl = leaderboard(res)
        cols = [c for c in _TABLE_COLUMNS if c in tbl.columns]
        say(f"  [{label}] 候选 {len(exprs)} -> 进池 {sub.size};样本外:")
        say(tbl[cols].to_string(float_format=lambda v: f"{v:+.4f}"))
        return {"label": label, "candidates": len(exprs), "pool_size": sub.size,
                "table": tbl, "members": [repr(m.expr) for m in sub.members],
                "pool_metrics": sub.test_ensemble()}

    def on_block(info: dict) -> None:
        i = info["index"]
        say(f"\n=== 第 {i} 块结束(至第 {info['generation']} 代)===")
        say(f"  封禁 {len(info['banned'])} 个一层结构,累计生效 {info['banned_total']} 个: "
            + ", ".join(info["banned"]))
        if info.get("exemplars_total"):
            say(f"  按 rank 相关封禁:本块新增样本 {info['exemplars_added']} 个,累计 {info['exemplars_total']} 个")
        if not (block_combine and combine_methods):
            return
        entry = {"index": i, "generation": info["generation"], "banned": list(info["banned"])}
        entry["block_only"] = score_archive(info["block_archive"], f"第{i}块 仅本块")
        if i > 1:                                   # block 1 alone IS the cumulative view
            entry["cumulative"] = score_archive(info["cumulative_archive"], f"第{i}块 累计1~{i}")
        block_reports.append(entry)

    ckpt_dir = None
    if out_root and checkpoint_every:
        ckpt_dir = Path(out_root) / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S")) / "checkpoints"
    if member is not None and gp_params and gp_params.get("exemplar_cache"):
        # ExemplarCache 一建就删掉同一根目录下别的视图:股票池的指纹放到自己的根目录,不然一开
        # 股票池挖掘(GP 或 RL)就把全市场那份缓存清掉了。根目录名带上掩码内容的哈希 —— 视图键
        # 只看日期和只数,同名名单换了成分会读到旧掩码下的指纹
        import hashlib
        tag = hashlib.sha1(np.packbits(member).tobytes() + "\n".join(view.symbols).encode()).hexdigest()[:12]
        gp_params = {**gp_params, "exemplar_cache": f"{gp_params['exemplar_cache']}_universe_{tag}"}
    if miner == "gp":
        gp = dict(population_size=24, max_depth=4, min_depth=2, top_k=None)
        gp.update(gp_params or {})
        gp.pop("rl_min_solo", None)                         # 只有 RL 的池子用
        # warmup 是 trim(下面显式传),调用方算不出 trim(warmed_view 在 :244,trim 只在
        # 落盘那两处报告里用,见 WP8 设计),所以不让它从 gp_params 里漏进来跟显式参数打架。
        gp.pop("warmup", None)
        seed_src = gp.pop("seed_population", None)
        if seed_src:
            from .legacy import load_seed_population
            gp["initial_population"] = load_seed_population(seed_src, n_minutes=view.n_minutes)
            say(f"GP seeds from {seed_src}: {len(gp['initial_population'])} parsed (strict grammar)")
        m = GPMiner(reg, view, target, pool_obj, seed=seed, device=device, min_coverage=min_coverage,
                    min_scored_days=min_scored_days,
                    chunk_symbols=chunk_symbols, chunk_days=chunk_days,
                    allow_boolean_ops=allow_boolean_ops, allow_resource_risk=allow_resource_risk,
                    skip_kernels=frozenset(skip_kernels), rolling_block=rolling_block, ban_top_n=ban_top_n,
                    on_block=on_block, cost_per_side=cost_per_side,
                    checkpoint_dir=ckpt_dir,
                    on_engine_progress=(engine_progress if engine_progress_every > 0 else None),
                    checkpoint_every=checkpoint_every, workdir=workdir, warmup=trim, **gp)
        if member is not None:
            # GPMiner 自己建引擎:不在这里套上股票池那层,GP 的面板就全是池外也有值的
            m.engine = UniverseEngine(m.engine, member)
        miner_ref.append(m)
        rep = m.mine(budget, on_generation=lambda g, n, s: progress(g, n, s))
        added, evaluated, engine_s = rep.added, rep.evaluated, rep.engine_seconds
        summary = f"{rep.generations} generations, archive {rep.archive_size}, best fitness {rep.best_fitness:+.4f}"
        rejected = rep.commit_reasons
        panels_of = m._panels
    else:
        if miner not in MINERS:
            raise ValueError(f"unknown miner {miner!r}; choose from {MINERS}")
        vocab = Vocabulary(reg, n_minutes=view.n_minutes, allow_boolean_ops=allow_boolean_ops,
                           allow_resource_risk=allow_resource_risk)
        # RL 和 GP 用同一个目标函数和同一套封禁 —— CLI 上的 --fitness /
        # --turnover-penalty / --ban / --ban-rank-corr / --ban-archive 走的是同一个
        # gp_params。以前 RL 的奖励是 Pearson 池 IC,和 gpu10 起 GP 用的
        # |rank IC| - 0.14 x 换手 不是一个东西,两条线挖出来的没法比。
        cfg = dict(gp_params or {})
        kind = str(cfg.get("fitness_kind", "ic"))
        penalty = float(cfg.get("turnover_penalty", 0.0))
        scorer = None
        if kind != "ic" or penalty > 0:
            from .mining.scoring import make_scorer
            scorer = make_scorer(target, kind, turnover_penalty=penalty, cost_per_side=cost_per_side,
                                 min_coverage=min_coverage, min_scored_days=min_scored_days,
                                 dates=list(view.dates), warmup=trim)
        rank_ban = None
        if cfg.get("ban_rank_corr"):
            from .mining.rankban import RankSimilarityBan
            rank_ban = RankSimilarityBan(target.shape, float(cfg["ban_rank_corr"]))
        banned_substructures = tuple(cfg.get("banned_substructures", ()))
        if banned_substructures:
            # 以前只有 GP 这条路(gp.py 的 GPMiner.__init__)验证过;RL 这边直接把元组
            # 原样塞给 BatchRewarder,一个打错的键在这里不报错、也不封任何东西 —— 静默
            # 失效。check_ban_key 和 GPMiner 用的是同一个函数,挑的字段集也一样
            # (registry.fields,不含分钟轴以外的东西)。惰性 import:env.py 不认识
            # gp.py,gp.py 也不认识 env.py,这里转一圈不会绕出循环。
            from .mining.gp import check_ban_key
            for key in banned_substructures:
                check_ban_key(key, reg, reg.fields)
        hooks = cfg.get("hooks")
        rew = BatchRewarder(engine, pool_obj, scorer=scorer, rank_ban=rank_ban, min_solo=cfg.get("rl_min_solo"),
                            banned_substructures=banned_substructures,
                            accept=(hooks.accept if hooks is not None else None),
                            panel_cache_items=int(cfg.get("panel_cache_items", 256)),
                            panel_cache_bytes=cfg.get("panel_cache_bytes", 8 * 1024 ** 3))
        if rank_ban is not None and cfg.get("ban_exemplars"):
            if cfg.get("exemplar_cache"):
                from .dailyfields import fingerprint
                from .mining.rankban import ExemplarCache, view_cache_key
                # daily= exactly as gp.py's own construction passes it (WP7 step 8).
                # Without it this branch keys the SAME daily-aware view differently
                # from the GP branch, and ExemplarCache._drop_other_views() rmtrees
                # whichever key it did not just open -- so a sweep alternating a GP
                # arm and an RL arm against one --exemplar-cache root re-pays the
                # whole ban-archive fingerprint install (1h-2h04m for 1049-1673
                # exemplars) on every switch, and the stale-fingerprint hazard this
                # keyword closes stays open on the RL half.
                rew.exemplar_cache = ExemplarCache(
                    cfg["exemplar_cache"],
                    view_cache_key(view, rank_ban.index.days.size, rank_ban.index.cols.size,
                                   daily=fingerprint(getattr(view.store, "daily", None))),
                    max_bytes=int(float(cfg.get("exemplar_cache_gb", 12.0)) * 1024 ** 3))
            n = rew.add_exemplars(list(cfg["ban_exemplars"]))
            say(f"rank-ban exemplars: {n} 个(来自 {len(cfg['ban_exemplars'])} 条表达式)")
        cls = {"alphagen": AlphaGenMiner, "alphaqcm": AlphaQCMMiner, "alphaforge": AlphaForgeMiner,
               "alphasage": AlphaSAGEMiner, "warmstart": PPOWarmStartQCMMiner}[miner]
        m = cls(vocab, rew, max_expr_len=max_expr_len, seed=seed, **(rl_params or {}))
        rep = m.mine(budget, on_progress=progress)
        added, evaluated, engine_s = rep.added, rep.evaluated, rep.engine_seconds
        summary = f"{rep.episodes} episodes, {rep.completed} completed expressions, best reward {rep.best_reward:+.4f}"
        rejected = rep.rejected
        panels_of = rew.panels
    say(f"done {time.time() - t0:.1f}s (engine {engine_s:.1f}s): {summary}; {evaluated} evaluations; "
        f"added {len(added)}; pool {pool_obj.size}")
    for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])[:8]:
        say(f"   x{v:<4d} {k}")
    if pool_obj.size == 0:
        say("pool is empty; nothing to report")
        return RunResult(None, 0, len(added), {}, [], None, time.time() - t0, engine_s, dict(rejected), summary)

    ens = pool_obj.test_ensemble()
    say("pool: " + "  ".join(f"{k} {v:+.4f}" for k, v in ens.items()))
    rows = []
    panels = panels_of(pool_obj.exprs())
    # NOT `realized = ...`: that rebinds the outer name to a trimmed copy, and the
    # final compare() below still passes full-length columns and target, so every
    # combination died on an out-of-bounds day index.
    realized_trim = np.asarray(pool_obj.realized, dtype=np.float64)[trim:]
    for i, (mem, p) in enumerate(zip(pool_obj.members, panels)):
        row = {"index": i, "expression": repr(mem.expr), "weight": float(pool_obj.weights[i])}
        if p is not None:
            pan = p[trim:].astype(float)
            row.update(ic_summary(pan, target[trim:]))
            # Return-based metrics per factor, not only for the pool and the
            # combinations: a high IC earned on a handful of days, or on a corner of
            # the universe, shows up here as a Sharpe that does not match. Same
            # function and same top_frac as the pool line, so the numbers are
            # directly comparable. Diagnostic only -- one holding period, no costs.
            bt = backtest_metrics(cs_pearson(pan, target[trim:]), pan, realized_trim,
                                  cost_per_side=cost_per_side, turnover_cap=turnover_cap)
            row.update({k: v for k, v in bt.items()
                        if k.startswith(("annualized_return", "sharpe", "max_drawdown", "turnover"))})
        rows.append(row)
        say(f"  [{i:2d}] w={row['weight']:+.4f} IC={row.get('ic_mean', float('nan')):+.4f}"
            f" sharpe={row.get('sharpe', float('nan')):+.2f}  {row['expression']}")

    table = None
    if combine_methods:
        from .combine.post_combine import compare, leaderboard
        # Costs reach the combinations too: without them this table (and every block
        # report) showed only the no-cost Sharpe while the model methods turned over
        # ~3x the book the simple weightings did -- 45-67% a day against 18-19% on
        # gpu5's first block -- which is exactly the ranking costs overturn.
        results, signs = compare(pool_obj.z_columns(), target, realized, methods=tuple(combine_methods),
                                 train_frac=combine_train_frac, align=True, seed=seed,
                                 cost_per_side=cost_per_side, turnover_cap=turnover_cap,
                                 smooth_halflives=smooth_halflives)
        table = leaderboard(results)
        cols = [c for c in _TABLE_COLUMNS if c in table.columns]
        say(f"combination (out-of-sample after {combine_train_frac:.0%} of days; {int((signs < 0).sum())} flipped):")
        say(table[cols].to_string(float_format=lambda v: f"{v:+.4f}"))

    run_dir = None
    if out_root:
        run_dir = Path(out_root) / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        (run_dir / "values").mkdir(parents=True, exist_ok=True)
        import pandas as pd
        if save_values:
            for i, p in enumerate(panels):
                if p is not None:
                    np.save(run_dir / "values" / f"{i:03d}.npy", p[trim:].astype(np.float32))
        pd.DataFrame(rows).to_csv(run_dir / "members.csv", index=False)
        np.save(run_dir / "combined.npy", pool_obj.combined()[trim:].astype(np.float32))
        if table is not None:
            table.to_csv(run_dir / "combination.csv")
        (run_dir / "meta.json").write_text(json.dumps({
            "store": str(getattr(store, "directory", store)),
            "store_contract_hash": getattr(store, "contract_hash", None),
            "start": start, "end": end, "dates": view.dates[trim:],
            "symbols": view.symbols, "label": lab.describe(), "miner": miner, "pool": pool, "budget": budget,
            "seed": seed, "pool_metrics": {k: float(v) for k, v in ens.items()}, "rejected": dict(rejected),
            "seconds": round(time.time() - t0, 1), "engine_seconds": round(engine_s, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"wrote {run_dir}")
    return RunResult(run_dir, pool_obj.size, len(added), {k: float(v) for k, v in ens.items()}, rows, table,
                     time.time() - t0, engine_s, dict(rejected), summary)


__all__ = ["run_mining", "RunResult", "make_label", "build_pool", "MINERS", "POOLS", "LABELS"]
