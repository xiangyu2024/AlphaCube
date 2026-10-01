"""挖完之后怎么把因子合成一个信号 —— 池内权重之外的、正经的组合层。

为什么需要它:池的 `_fit_weights` 是**搜索过程中的奖励信号**,要求每插入一条
就能重算一次,所以它必须便宜、粗糙。它不是、也不该是最终的组合方法。但
上一版的训练脚本直接把池权重当成果输出了,于是出现两个具体问题:

1. **方向没对齐。** GP 默认 `fitness_signed=False`,即用 `|IC|` 筛选 ——
   明确承认"负 IC 因子是可反转的信号,和它的正镜像一样有价值"。但从来没有
   任何地方真的去反转它。于是池里同时存在 IC=-0.015 和 IC=+0.010 的成员,
   组合时互相抵消。筛选口径和组合口径不一致,是实打实的缺陷。
2. **权重经常退化成等权。** 上一版项目 `MseAlphaPool._fit_weights` 的 OLS 前置条件是
   `np.all(np.isfinite(design), axis=1)` —— 要求**所有成员在同一格子上同时
   有限**。只要有一条成员是常数/布尔型(z-score 后整片 NaN),这个交集就塌了,
   直接走 `return np.full(n, 1/n)` 兜底。输出里权重全是 +0.1000、
   `combined_coverage` 很低,就是这个症状。(本项目 `mining.pool.MsePool` 把缺失 z-score 记 0,
   不再这样塌;但池权重仍只是搜索期的奖励信号,不是最终组合。)

本模块做的事:先按各因子自身 IC 的符号**对齐方向**,再用多种方法组合,并且
**统一在样本外比较** —— 权重/模型一律只用训练段拟合,指标只报测试段。不这样
做的话,树模型在样本内必然好看,但那个数字没有意义。

方法:
    equal      等权(方向对齐后)
    ic         按训练段 |IC| 加权
    ric        按训练段 |rank IC| 加权 —— 和挖掘期的适应度同口径
    icir       按训练段 |ICIR| 加权 —— 稳定性比绝对强度更值钱
    lstsq      OLS(池当前用的口径,作为基线)
    ridge      带正则的线性,因子间共线时比 OLS 稳
    gbdt       HistGradientBoosting(sklearn 自带,原生吃 NaN)
    lightgbm   需要 pip install lightgbm
    xgboost    需要 pip install xgboost
    roll_ic    滚动 IC 加权:每 20 天用前 250 天的逐日 IC 均值重算权重(带符号)
    roll_ric   滚动 rank IC 加权,同上但用逐日 rank IC
    roll_lstsq 滚动 OLS:每 20 天用前 250 天重新拟合
    roll_ridge 滚动岭回归
    factorgcl  超图 GNN(`combine/factorgcl/`):一天一张图、股票为节点、节点
               特征是该股当天的因子向量,在先验因子超边之外再学一组隐超边,
               训练目标是 MSE(前瞻收益) + 时序残差对比损失。它和树模型的
               区别在于**显式建模同一天股票之间的关系**,而不是把每个
               (股票,天) 当独立样本

滚动方法(`roll_*`)和上面几种的区别不在算法,在**什么时候拟合**:静态方法把训练段
一次拟合出来的权重原样用到三年以后,滚动方法每 20 个交易日用最近 250 天重来一次,
第 t 天的权重只看得到 t 之前的数据。前 250 天没有历史可用,留 NaN(落在训练段里,
不进测试段指标)。注意比较时的口径差异:静态方法只用训练段的信息,滚动方法在测试段
里也在持续用"当天之前"的信息重拟合 —— 这不是前视,但它确实比静态方法更新。

树模型的样本量:全市场 10 年 x 10 因子 = 1.26 亿行,直接喂进去不现实,所以
拟合前按 `max_fit_rows` 随机下采样(预测仍在全量上做)。

平滑再交易(`smooth_halflives`):每种方法合成出信号后,再试一遍"交易它的逐股 EMA"。
半衰期只在训练段上按扣成本夏普选(0 = 不平滑,永远在候选里),测试段报这个选择的
结果:`halflife` / `sharpe_smooth` / `turnover_smooth`。gpu5~gpu9 的组合无成本夏普
4~5,但日换手 40~70%,15bp 吃掉绝大部分 —— 平滑是在不动因子的前提下压换手。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..metrics import backtest_metrics, cross_sectional_pearson, ema_rows, rank_rows

WEIGHT_METHODS = ("equal", "ic", "ric", "icir", "lstsq", "ridge")
MODEL_METHODS = ("ridge", "gbdt", "lightgbm", "xgboost", "factorgcl")
ROLLING_METHODS = ("roll_ic", "roll_ric", "roll_lstsq", "roll_ridge")
DEFAULT_METHODS = ("equal", "ic", "icir", "lstsq", "gbdt")

PREDICT_CELLS = 20_000_000    # 分块预测时一块最多这么多个格子(n 个因子 -> 行数 = 它 / n)
ROLL_WINDOW = 250      # 回看一年
ROLL_STEP = 20         # 每月重拟合一次
RIDGE_ALPHA = 1e-2     # 相对 trace(XᵀX)/n 的正则强度


def factor_ic_series(column: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐日截面 IC 序列。"""
    return cross_sectional_pearson(column, target)


def align_signs(
    columns: Sequence[np.ndarray], target: np.ndarray, days: Optional[slice] = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """把每个因子翻成"IC 为正"的方向,返回 (对齐后的列, 符号)。

    符号只用 `days` 指定的训练段估计 —— 用全样本定方向就是在偷看测试段,
    那会把一个前视偏差伪装成组合方法的收益。IC 非有限(常数/全 NaN 因子)时
    符号取 +1,不翻。
    """
    signs = np.ones(len(columns), dtype=np.float64)
    for i, column in enumerate(columns):
        sub_c = column[days] if days is not None else column
        sub_t = target[days] if days is not None else target
        ic = float(np.nanmean(factor_ic_series(sub_c, sub_t)))
        if np.isfinite(ic) and ic < 0:
            signs[i] = -1.0
    # 没翻的列直接复用,不复制:块边界上 30 条 (1942, 5220) float64 是 2.4 GB,
    # `s * c` 会把其中一半白白再存一份。调用方不改这些数组(本模块只读它们)。
    return [c if s > 0 else -c for s, c in zip(signs, columns)], signs


def _weights_for(
    method: str, columns: Sequence[np.ndarray], target: np.ndarray, days: slice
) -> np.ndarray:
    n = len(columns)
    if method == "equal":
        return np.full(n, 1.0 / n, dtype=np.float64)

    if method in ("ic", "ric", "icir"):
        scores = np.zeros(n, dtype=np.float64)
        # "ric" 用逐日 rank IC —— 目标的秩只算一次,不是每个因子重算一遍
        tgt = rank_rows(target[days]) if method == "ric" else target[days]
        for i, column in enumerate(columns):
            col = rank_rows(column[days]) if method == "ric" else column[days]
            series = factor_ic_series(col, tgt)
            finite = series[np.isfinite(series)]
            if finite.size < 2:
                continue
            mean = float(finite.mean())
            if method in ("ic", "ric"):
                scores[i] = abs(mean)
            else:
                std = float(finite.std())
                scores[i] = abs(mean / std) if std > 1e-12 else 0.0
        total = scores.sum()
        # 全零(所有因子都退化)时退回等权,并且这是**显式**的,不是静默兜底
        return scores / total if total > 1e-12 else np.full(n, 1.0 / n)

    if method == "lstsq":
        # A missing value is a neutral 0, as in the pool's own fit and in the
        # combination below. Requiring every member finite in a row -- the rule
        # the pool dropped for exactly this reason -- leaves no rows at all once
        # the pool is large: 145 members on gpu5's first block, and every
        # weight method fell back to 1/n.
        design = np.nan_to_num(np.stack([c[days] for c in columns], axis=-1).reshape(-1, n), nan=0.0)
        y = target[days].reshape(-1)
        mask = np.isfinite(y)
        if mask.sum() < max(n, 10):
            return np.full(n, 1.0 / n, dtype=np.float64)
        weights, *_ = np.linalg.lstsq(design[mask], y[mask], rcond=None)
        return weights.astype(np.float64)

    raise ValueError(f"{method} 不是权重类方法")


def _daily_ic_matrix(
    columns: Sequence[np.ndarray], target: np.ndarray, rank: bool
) -> np.ndarray:
    """(n_factors, T) 的逐日 IC / rank IC 矩阵,滚动权重是它的窗口均值。

    整段算一次,窗口里只做平均。否则每次重拟合都要重排一遍窗口内的秩 ——
    100 个因子 x 85 次重拟合 = 8500 次 250x5220 的排序,二十多分钟;这样是 100 次。
    """
    tgt = rank_rows(target) if rank else target
    rows = [factor_ic_series(rank_rows(c) if rank else c, tgt) for c in columns]
    return np.asarray(rows, dtype=np.float64)


def _daily_moments(
    columns: Sequence[np.ndarray], target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """逐日的 (XᵀX, Xᵀy)。窗口回归的正规方程 = 窗口内这些矩阵直接相加。

    (T, n, n) 对 100 个因子是 155 MB,换来的是每次重拟合只加 250 个小矩阵,
    而不是重新扫一遍 130 万行的设计矩阵。缺失值照池的口径记 0。
    """
    T = target.shape[0]
    n = len(columns)
    gram = np.zeros((T, n, n), dtype=np.float64)
    rhs = np.zeros((T, n), dtype=np.float64)
    for t in range(T):
        y = target[t]
        mask = np.isfinite(y)
        if not mask.any():
            continue
        x = np.nan_to_num(np.stack([c[t] for c in columns], axis=-1)[mask], nan=0.0)
        gram[t] = x.T @ x
        rhs[t] = x.T @ y[mask]
    return gram, rhs


def _solve_normal(gram: np.ndarray, rhs: np.ndarray, ridge_alpha: float) -> np.ndarray:
    n = gram.shape[0]
    if ridge_alpha > 0:
        scale = float(np.trace(gram)) / max(n, 1)
        gram = gram + (ridge_alpha * scale) * np.eye(n)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def rolling_combination(
    method: str,
    columns: Sequence[np.ndarray],
    target: np.ndarray,
    window: int = ROLL_WINDOW,
    step: int = ROLL_STEP,
    ridge_alpha: float = RIDGE_ALPHA,
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """滚动组合:第 t 天的权重只用 [t-window, t) 拟合,每 `step` 天重拟合一次。

    返回 (合成面板, [(生效首日, 权重), ...])。权重历史本身就是结论之一:
    它到底变不变 —— 不变的话滚动就只是把静态方法跑贵了。
    """
    if method not in ROLLING_METHODS:
        raise ValueError(f"{method} 不是滚动类方法")
    T, N = target.shape
    n = len(columns)
    if window < 2 or window >= T:
        raise ValueError(f"滚动窗口 {window} 对 {T} 天的样本不合法")
    if step < 1:
        raise ValueError("step 必须 >= 1")

    ic_mat = gram = rhs = None
    if method in ("roll_ic", "roll_ric"):
        ic_mat = _daily_ic_matrix(columns, target, rank=(method == "roll_ric"))
    else:
        gram, rhs = _daily_moments(columns, target)

    out = np.full((T, N), np.nan, dtype=np.float64)
    history: List[Tuple[int, np.ndarray]] = []
    for start in range(window, T, step):
        lo = start - window
        if ic_mat is not None:
            block = ic_mat[:, lo:start]
            finite = np.isfinite(block)
            counts = finite.sum(axis=1)
            sums = np.where(finite, block, 0.0).sum(axis=1)
            # 带符号:权重自己管方向,某个因子在最近一年反了就该反着用
            scores = np.where(counts >= 2, sums / np.maximum(counts, 1), 0.0)
            total = float(np.abs(scores).sum())
            weights = scores / total if total > 1e-12 else np.full(n, 1.0 / n)
        else:
            weights = _solve_normal(gram[lo:start].sum(axis=0), rhs[lo:start].sum(axis=0),
                                    ridge_alpha if method == "roll_ridge" else 0.0)
        history.append((start, weights))

        stop = min(start + step, T)
        seg = slice(start, stop)
        acc = np.zeros((stop - start, N), dtype=np.float64)
        seen = np.zeros((stop - start, N), dtype=bool)
        for column, weight in zip(columns, weights):
            values = column[seg]
            finite = np.isfinite(values)
            acc += weight * np.where(finite, values, 0.0)
            seen |= finite
        acc[~seen] = np.nan
        out[seg] = acc
    return out, history


def _model_prediction(
    method: str,
    columns: Sequence[np.ndarray],
    target: np.ndarray,
    days: slice,
    max_fit_rows: int,
    seed: int,
) -> np.ndarray:
    """用训练段拟合一个模型,在全样本上预测,返回 (T, N) 面板。"""
    from ..models.estimators import ESTIMATOR_REGISTRY

    key = {"gbdt": "hgb"}.get(method, method)   # 注册表里叫 hgb
    if key not in ESTIMATOR_REGISTRY:
        raise KeyError(
            f"估计器 '{key}' 不可用。已注册: {sorted(ESTIMATOR_REGISTRY)}。"
            "lightgbm/xgboost 是可选依赖,需要单独 pip install。"
        )
    T, N = target.shape
    n = len(columns)
    # Never materialise the whole (T*N, n) design matrix: 30 members of the
    # production cube is 2.4 GB in float64 and 100 is 8.1 GB, allocated at the
    # block boundary on top of everything the search is already holding. The fit
    # takes only the sampled rows, and the prediction goes chunk by chunk.
    flat = [c.reshape(-1) for c in columns]

    train = np.zeros((T, N), dtype=bool)
    train[days] = True
    y = target.reshape(-1)
    fit_rows = np.flatnonzero(train.reshape(-1) & np.isfinite(y))
    if fit_rows.size < 100:
        raise ValueError("训练段有限样本不足 100 行,无法拟合模型")
    if fit_rows.size > max_fit_rows:
        rng = np.random.default_rng(seed)
        fit_rows = rng.choice(fit_rows, size=max_fit_rows, replace=False)

    # Seeded: HistGradientBoosting turns on early stopping above 10k rows and draws
    # its validation split at random, so an unseeded "gbdt" gave a different
    # combination on every call -- out-of-sample Sharpe 5.0 then 3.0 for the same
    # inputs and seed on the test fixture, and the same on the production blocks.
    try:
        model = ESTIMATOR_REGISTRY[key](random_state=seed)
    except TypeError:                                  # an estimator without a seed
        model = ESTIMATOR_REGISTRY[key]()
    X_fit = np.empty((fit_rows.size, n), dtype=np.float64)
    for j, column in enumerate(flat):
        X_fit[:, j] = column[fit_rows]
    model.fit(X_fit, y[fit_rows])
    del X_fit

    out = np.empty(T * N, dtype=np.float64)
    step = max(1, PREDICT_CELLS // max(n, 1))
    for lo in range(0, T * N, step):
        hi = min(lo + step, T * N)
        chunk = np.empty((hi - lo, n), dtype=np.float64)
        for j, column in enumerate(flat):
            chunk[:, j] = column[lo:hi]
        out[lo:hi] = model.predict(chunk)
    return out.reshape(T, N)


def _factorgcl_prediction(
    columns: Sequence[np.ndarray],
    target: np.ndarray,
    days: slice,
    seed: int,
    epochs: int = 5,
    **kwargs,
) -> np.ndarray:
    """超图 GNN 组合:一天一张图,节点=股票,节点特征=该股当天的因子向量。

    列直接作为因子面板喂给组合器 —— 它的 `build_prior_incidence` 只是把每列
    逐日 z-score 后堆成先验关联矩阵(对已经是 z-score 的列是幂等的),不再
    像上一版那样逐条走 AST 重算。
    `train_days` 保证优化器只看得到训练段。
    """
    from .factorgcl import FactorGCLCombiner

    combiner = FactorGCLCombiner(seed=seed, **kwargs)
    combiner.fit(columns, target, epochs=epochs, train_days=days)
    return combiner.predict(columns)


@dataclass
class CombinationResult:
    method: str
    weights: Optional[np.ndarray]
    combined: np.ndarray
    metrics: Dict[str, float]


def combine(
    method: str,
    columns: Sequence[np.ndarray],
    target: np.ndarray,
    real_returns: np.ndarray,
    train_days: slice,
    test_days: slice,
    max_fit_rows: int = 2_000_000,
    seed: int = 0,
    factorgcl_kwargs: Optional[Dict] = None,
    cost_per_side: float = 0.0,
    turnover_cap: Optional[float] = None,
    smooth_halflives: Sequence[float] = (),
    roll_window: int = ROLL_WINDOW,
    roll_step: int = ROLL_STEP,
) -> CombinationResult:
    """按 `method` 组合,指标只在 `test_days`(样本外)上算。

    `cost_per_side` / `turnover_cap` 原样交给 `backtest_metrics`,多出
    `*_net`(扣成本)和 `*_capped`(限换手)两套指标。不传就只有无成本那套 ——
    模型类方法的日换手常是简单加权的 3 倍,只看无成本会把排名看反。"""
    if method == "factorgcl":
        combined = _factorgcl_prediction(columns, target, train_days, seed,
                                         **(factorgcl_kwargs or {}))
        weights = None
    elif method in ROLLING_METHODS:
        # 滚动方法自己定权重,`train_days` 只用来选平滑半衰期
        combined, history = rolling_combination(method, columns, target, roll_window, roll_step)
        weights = history[-1][1] if history else None
    elif method in MODEL_METHODS:
        combined = _model_prediction(method, columns, target, train_days, max_fit_rows, seed)
        weights = None
    if method == "factorgcl" or method in ROLLING_METHODS or method in MODEL_METHODS:
        # 模型对一个成员都没值的格子也会吐出一个数(特征全按缺失喂进去);加权法在那里是 NaN,
        # 这里对齐 —— 否则股票池挖掘的样本外书里会混进池外的票
        any_seen = np.zeros(np.shape(target), dtype=bool)
        for column in columns:
            any_seen |= np.isfinite(column)
        combined = np.where(any_seen, combined, np.nan)
    else:
        weights = _weights_for(method, columns, target, train_days)
        # A member's missing value counts as neutral 0 (the pool's convention); a
        # cell is missing only when every member is. Summing NaNs instead made the
        # whole combination NaN wherever ANY member was: 3.5% of cells with 30
        # members, and every cell with the 145 of gpu5's first block.
        combined = np.zeros_like(target, dtype=np.float64)
        seen = np.zeros(target.shape, dtype=bool)
        for column, weight in zip(columns, weights):
            finite = np.isfinite(column)
            combined += weight * np.where(finite, column, 0.0)
            seen |= finite
        combined[~seen] = np.nan

    ic_series = cross_sectional_pearson(combined[test_days], target[test_days])
    metrics = dict(backtest_metrics(ic_series, combined[test_days], real_returns[test_days],
                                    cost_per_side=cost_per_side, turnover_cap=turnover_cap))
    metrics["rank_ic"] = float(np.nanmean(cross_sectional_pearson(
        rank_rows(combined[test_days]), rank_rows(target[test_days]))))
    metrics["coverage"] = float(np.isfinite(combined[test_days]).mean())
    if method in ROLLING_METHODS:
        metrics["refits"] = float(len(history))
    if smooth_halflives:
        metrics.update(smoothed_metrics(combined, target, real_returns, train_days, test_days,
                                        smooth_halflives, cost_per_side))
    return CombinationResult(method, weights, combined, metrics)


def smoothed_metrics(
    combined: np.ndarray,
    target: np.ndarray,
    real_returns: np.ndarray,
    train_days: slice,
    test_days: slice,
    halflives: Sequence[float],
    cost_per_side: float = 0.0,
) -> Dict[str, float]:
    """交易合成信号的逐股 EMA(`metrics.ema_rows`,因果,不看未来)而不是信号本身。

    半衰期只用训练段选:每个候选(外加 0 = 不平滑)在训练段上算十分位多空书的
    夏普 —— 有成本时是扣成本的 —— 取最高的;测试段只报这一个选择。EMA 在整段上
    递推,所以测试段第一天接着训练段的状态,这不是前视:第 t 天只用到 t 及以前。"""
    net_key = "sharpe_net" if cost_per_side > 0 else "sharpe"
    no_ic = np.zeros(0)
    best_h, best_s = 0.0, -np.inf
    for h in sorted({0.0, *(float(x) for x in halflives)}):
        sm = ema_rows(combined, h)
        s = backtest_metrics(no_ic, sm[train_days], real_returns[train_days],
                             cost_per_side=cost_per_side).get(net_key, np.nan)
        if np.isfinite(s) and s > best_s:
            best_h, best_s = h, float(s)
    sm = ema_rows(combined, best_h)
    t = backtest_metrics(cross_sectional_pearson(sm[test_days], target[test_days]), sm[test_days],
                         real_returns[test_days], cost_per_side=cost_per_side)
    return {"halflife": best_h, "sharpe_smooth": float(t.get(net_key, np.nan)),
            "turnover_smooth": float(t.get("turnover", np.nan)),
            "annualized_return_smooth": float(t.get("annualized_return_net" if cost_per_side > 0
                                                    else "annualized_return", np.nan))}


def compare(
    columns: Sequence[np.ndarray],
    target: np.ndarray,
    real_returns: np.ndarray,
    methods: Sequence[str] = DEFAULT_METHODS,
    train_frac: float = 0.7,
    align: bool = True,
    max_fit_rows: int = 2_000_000,
    seed: int = 0,
    factorgcl_kwargs: Optional[Dict] = None,
    cost_per_side: float = 0.0,
    turnover_cap: Optional[float] = None,
    smooth_halflives: Sequence[float] = (),
    roll_window: int = ROLL_WINDOW,
    roll_step: int = ROLL_STEP,
) -> Tuple[List[CombinationResult], np.ndarray]:
    """跑一组组合方法,全部在同一个样本外切分上比较。

    返回 (结果列表, 方向符号)。`align=False` 可用来对照"不对齐方向"的效果 ——
    这正是当前池权重的口径。
    """
    if not columns:
        raise ValueError("没有因子可组合")
    T = target.shape[0]
    split = max(1, min(T - 1, int(T * train_frac)))
    train_days, test_days = slice(0, split), slice(split, T)

    if align:
        columns, signs = align_signs(columns, target, days=train_days)
    else:
        signs = np.ones(len(columns), dtype=np.float64)

    results: List[CombinationResult] = []
    for method in methods:
        try:
            results.append(combine(method, columns, target, real_returns,
                                   train_days, test_days, max_fit_rows, seed,
                                   factorgcl_kwargs, cost_per_side, turnover_cap,
                                   smooth_halflives, roll_window, roll_step))
        except Exception as exc:  # 缺可选依赖/样本不足,跳过但要说出来
            results.append(CombinationResult(
                method, None, np.full_like(target, np.nan, dtype=np.float64),
                {"error": str(exc)[:120]}))
    return results, signs


def leaderboard(results: Sequence[CombinationResult]) -> "object":
    """结果 → 按样本外 ICIR 排序的表。"""
    import pandas as pd

    rows = []
    for result in results:
        row = {"method": result.method}
        row.update({k: v for k, v in result.metrics.items() if k != "error"})
        if "error" in result.metrics:
            row["error"] = result.metrics["error"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")
    if "icir" in table.columns:
        table = table.sort_values("icir", ascending=False, na_position="last")
    return table


__all__ = [
    "CombinationResult", "DEFAULT_METHODS", "MODEL_METHODS", "ROLLING_METHODS",
    "ROLL_STEP", "ROLL_WINDOW", "WEIGHT_METHODS",
    "align_signs", "combine", "compare", "factor_ic_series", "leaderboard",
    "rolling_combination", "smoothed_metrics",
]
