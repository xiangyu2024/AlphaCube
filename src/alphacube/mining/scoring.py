"""一份目标函数,GP 和 RL 共用。

gpu10 起 GP 的适应度是 ``|rank IC| - λ x 十分位多空书的日换手``(λ=0.14 是
15 bp/边换算到 rank IC 的单位),而 RL 的奖励还停在 Pearson |IC| 上 ——
`env.BatchRewarder` 里就是 `nanmean_quiet(cs_pearson(...))`。这不是口径上的
小分歧:在跨版本的 2775 个健康因子上量过,|IC| 对扣费净夏普的 Spearman 只有
+0.14,|rank IC| 是 +0.45,换手是 -0.76,而 ``|rank IC| - 0.14 x 换手`` 是
**+0.89**。两条搜索线各用各的目标,挖出来的东西没法比,RL 那条还会把一天的
GPU 花在扣完成本归零的高换手因子上(gpu5~gpu9 反复发生过)。

这里只搬"带换手惩罚的那条路"(GP 的 ``_rich`` 分支)。GP 原来的纯 |IC| 批量
路径留在 `gp.py` 里不动 —— 它按整批算 `batched_ic`,是另一种形状的接口。

2026-10-03 加了第二份目标(``LongScorer``,``--fitness long``),上面那份原样留着。
``|rank IC| - 0.14 x 换手`` 奖励的是整条截面的排序,而这套因子的多空差价八成在空头腿
(后 10% 日均 -23.5bp,前 3% 才 +8~+10bp):A 股空头做不了,组合是纯多头。新目标直接按
"单独拿这个因子做纯多头"打分,见 `LongScorer`。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..metrics import (MIN_PAIRS, _avg_ranks_t, _book_turnover_t, _cs_corr_days_t, nanmean_quiet,
                       rank_ic_turnover, torch_rank_rows)

FITNESS_KINDS = ("ic", "rank_ic", "long", "long_hit")
BOOK_TOP_FRAC = 0.1          # 十分位多空书 —— 报告里交易的那一个


class PanelScorer:
    """一个面板 -> {ic, rank_ic, turnover, fit}。

    门槛和池、和 GP 的 IC 路径一致:分数有限、覆盖率、以及"有多少天算得出
    Pearson IC"的占比(LinearPool 就是这么取的)。不满足时 fit = -inf,
    而不是一个小分数 —— 这是拒绝,不是惩罚。
    """

    def __init__(self, target: np.ndarray, *, kind: str = "rank_ic", turnover_penalty: float = 0.14,
                 signed: bool = False, top_frac: float = BOOK_TOP_FRAC,
                 min_coverage: float = 0.5, min_scored_days: float = 0.5) -> None:
        if kind not in ("ic", "rank_ic"):
            raise ValueError(f"kind must be ic or rank_ic, not {kind!r} (纯多头口径用 LongScorer / make_scorer)")
        if turnover_penalty < 0:
            raise ValueError("turnover_penalty must be >= 0")
        self.kind = kind
        self.turnover_penalty = float(turnover_penalty)
        self.signed = bool(signed)
        self.top_frac = float(top_frac)
        self.min_coverage = float(min_coverage)
        self.min_scored_days = float(min_scored_days)
        self.target = np.ascontiguousarray(np.asarray(target, dtype=np.float64))
        self.finite = np.isfinite(self.target)
        self.cells = max(int(self.finite.sum()), 1)
        self.usable_days = max(int((self.finite.sum(axis=1) >= MIN_PAIRS).sum()), 1)
        self.target_t64 = torch.from_numpy(self.target)
        self.target_ranks_t = torch_rank_rows(self.target)

    def score(self, panel: np.ndarray, gated: bool = True) -> Dict[str, float]:
        """``gated=False``:不套覆盖率 / 可算天数这两道拒绝门槛 —— 那是给单个候选的。
        池子的合成信号已经由成员各自过了门槛,它的覆盖率按 z 分数算(逐日标准化后,
        截面没有离散度的天是 NaN),会比成员的原始覆盖率低,不该因此拿 -inf。"""
        ric_days, turnover = rank_ic_turnover(panel, self.target_ranks_t, self.top_frac)
        x = torch.from_numpy(np.ascontiguousarray(panel)).to(torch.float64)
        ic_days = _cs_corr_days_t(x, self.target_t64).numpy()
        ic, ric = nanmean_quiet(ic_days), nanmean_quiet(ric_days)
        base = ric if self.kind == "rank_ic" else ic
        cov = float((np.isfinite(panel) & self.finite).sum() / self.cells)
        scored = np.isfinite(ic_days).sum() / self.usable_days
        fit = -np.inf
        if np.isfinite(base) and (not gated or (cov >= self.min_coverage
                                                and (self.min_scored_days <= 0 or scored >= self.min_scored_days))):
            fit = (base if self.signed else abs(base)) - self.turnover_penalty * turnover
        return {"ic": ic, "rank_ic": ric, "turnover": float(turnover), "fit": float(fit)}

    def score_combined(self, panel: np.ndarray) -> Dict[str, float]:
        """池子的合成信号:不套单因子的拒绝门槛(见 `score` 的 gated)。"""
        return self.score(panel, gated=False)


# --- 纯多头口径 ---------------------------------------------------------------

LONG_SIDE_FRAC = 0.5         # "多头端":按因子方向排在前一半的票,多头 IC / rank IC 只在这些票上算
LONG_MIN_DISTINCT = 0.5      # 每天"不同取值数 / 有值的票数"的中位数下限,见 LongScorer
LONG_MAX_BOUNDARY_TIES = 0.1 # 书里"和书外的票并列、靠代码顺序才进来"的仓位占比(日均)上限,见 LongScorer
LONG_MAX_CODE_ORDER = 0.5    # 书附近(排在最前 2 x top_frac 的位置)"排名 vs 股票列号"的日均相关上限,见 LongScorer
# 四个分量各自除以它的尺度再按权重平均。尺度 = 现有因子库(挖掘 + 手写)上该分量"90 分位 - 中位数"的大小,
# 所以每个分量的 1 个单位都是"比中等因子好到前 10% 的那段距离"。数是 scripts/calibrate_long_reward.py
# 在训练段前一半、1200 个随机抽的因子上量出来再取整的(2026-10-03:2.19 / 0.137 / 0.0052 / 0.0122)。
# 权重:同一份校准里,用前一半的奖励去排、看后一半单独做纯多头的扣费夏普 —— 多头端 rank IC 最能认出
# "平滑以后好用"的因子(Spearman +0.77),夏普 / 年化最能认出"原样就好用"的(+0.90 / +0.86),
# 多头端 Pearson IC 最弱(+0.2,被极端值带着走),所以给半份。这组权重挑出的前 10% 因子,后一半夏普中位数
# +1.44(原样)/ +1.57(半衰期 5 平滑);老适应度 |rank IC| - 0.14 x 换手 挑出来的是 +0.71 / +1.33。
# LONG_UNIT 把结果放回老适应度的量级(因子库里 99 分位约 0.03),RL 的无效表达式惩罚(-1)不用跟着改。
LONG_SCALES = {"sharpe": 2.0, "ann": 0.13, "ic": 0.005, "rank_ic": 0.012}
LONG_WEIGHTS = {"sharpe": 1.0, "ann": 1.0, "ic": 0.5, "rank_ic": 1.5}
LONG_UNIT = 0.03
LONG_COLUMNS = ("long_sign", "long_sharpe", "long_ann", "long_turnover", "long_ic", "long_rank_ic", "long_tie",
                "long_code", "distinct")


class LongScorer:
    """一个面板 -> 单独拿它做纯多头有多好。接口和 `PanelScorer` 一样,GP 和 RL 共用。

    书:每天等权持有因子值最高的 ``top_frac``(方向 +1)或最低的 ``top_frac``(方向 -1),权重和 1;
    当天收益减去"当天所有有标签的票"的等权收益(基准对每个候选都一样 —— 用"有因子值的票"做基准的话,
    候选在自己不持有的好票上给 NaN 就能把基准压低);换手 ``sum|w_t - w_t-1|``(第一天从空仓建仓)每单位付
    ``cost_per_side``。标签缺的持仓(次日开盘封涨停、买不进)当天记 0;从第一次有持仓起每天都算,
    因子整天没值的日子就是空仓跑输基准、清仓的费用照付。四个分量:

      long_sharpe   这本书日扣费超额的年化夏普
      long_ann      日扣费超额 x 252
      long_ic       多头端(按方向排在前 ``side_frac`` 的票)上因子和标签的日 Pearson 相关的均值
      long_rank_ic  同一批票上两边秩的相关(秩是全截面的秩,和 rank IC 一个口径)

    适应度 = LONG_UNIT x 加权平均(分量 / 尺度)。两个方向各算一遍、取高的那个(``long_sign`` 记下是哪个):
    因子的符号是自由的,要找的是"哪一头拿来做多头好用"。合成信号的方向是定的,`score_combined` 只算 +1。

    比老目标多两道拒绝门槛,都是因为十分位书在并列上按股票代码顺序破平局(老目标用秩相关,并列拿不到分,
    所以以前不需要):
      ``min_distinct``         一个常数因子的"前 10%"就是代码最靠后的那批票,换手为 0 —— 不设门槛搜索会找到它;
      ``max_boundary_ties``    取值不少、但头部被截平的因子(clip / sigmoid 饱和 / 把最高的一批换成同一个数):
                               书的边界落在一段并列里,进书的是这段并列里代码靠后的。这样进来的仓位日均超过
                               这个比例,这个方向就不算(``long_tie`` 记的是这个比例)。
      ``max_code_order``       cs_rank / nt_cs_rank 自己按代码顺序破平局、吐出来的值互不相同,上面两道都看不见:
                               ``cs_rank(一个截面常数)`` 就是列号本身。所以再量一个 ``long_code``:排在最前
                               2 x top_frac 的位置上,"排名"和"股票列号"的日均相关;绝对值超过上限这个方向不算。
                               (2026-10-04 在真实训练段标签上量过:纯列号的"因子"两个方向夏普都是负的,
                               适应度 +0.001,所以 gp_long_1 / rl_long_1 没有这道门槛也没被它带偏。)
    后两道对池子的合成信号也生效(`score_combined`):一两个成员的小池子可能把成员并列的那一头拿来做多。

    多头端的 IC / rank IC 不含费用,权重合起来占一半:一个换手偏高、单独做书扣完费是亏的因子,只要多头端排序准,
    适应度仍然可以是正的。这是有意的 —— 校准里多头端 rank IC 最能认出"平滑以后好用"的因子,而组合层本来就平滑。
    ``distinct`` 用的是 torch.median(偶数天时取中间两个里小的那个)。
    """

    KIND = "long"
    MERGE_WEIGHTS = True
    DEFAULT_WEIGHTS = LONG_WEIGHTS
    DEFAULT_SCALES = LONG_SCALES
    COMPONENTS = tuple(LONG_WEIGHTS)
    columns = LONG_COLUMNS                 # GP 存档里多写的列

    def __init__(self, target: np.ndarray, *, cost_per_side: float = 0.0015, top_frac: float = BOOK_TOP_FRAC,
                 side_frac: float = LONG_SIDE_FRAC, min_coverage: float = 0.5, min_scored_days: float = 0.5,
                 min_distinct: float = LONG_MIN_DISTINCT, max_boundary_ties: float = LONG_MAX_BOUNDARY_TIES,
                 max_code_order: float = LONG_MAX_CODE_ORDER,
                 weights: Optional[Dict[str, float]] = None,
                 scales: Optional[Dict[str, float]] = None, unit: float = LONG_UNIT) -> None:
        if cost_per_side < 0:
            raise ValueError("cost_per_side must be >= 0")
        if not 0 < top_frac <= side_frac <= 1:
            raise ValueError("need 0 < top_frac <= side_frac <= 1")
        self.kind = self.KIND
        self.cost_per_side = float(cost_per_side)
        self.top_frac = float(top_frac)
        self.side_frac = float(side_frac)
        self.min_coverage = float(min_coverage)
        self.min_scored_days = float(min_scored_days)
        self.min_distinct = float(min_distinct)
        self.max_boundary_ties = float(max_boundary_ties)
        self.max_code_order = float(max_code_order)
        # 第一版只有四个分量,给哪个改哪个;第二版分量是一张菜单,给了 weights 就是"只用这几个"
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})} if self.MERGE_WEIGHTS or weights is None \
            else dict(weights)
        self.scales = {**self.DEFAULT_SCALES, **(scales or {})}
        if not set(self.weights) <= set(self.scales) or not set(self.weights) <= set(self.COMPONENTS):
            raise ValueError(f"weights 的键只能是 {sorted(self.COMPONENTS)},而且每个都要有 scale")
        if min(self.weights.values()) < 0 or sum(self.weights.values()) <= 0 or min(self.scales.values()) <= 0:
            raise ValueError("weights 要非负且不全为 0,scales 要为正")
        self.unit = float(unit)
        self.target = np.ascontiguousarray(np.asarray(target, dtype=np.float64))
        self.finite = np.isfinite(self.target)
        self.cells = max(int(self.finite.sum()), 1)
        self.usable_days = max(int((self.finite.sum(axis=1) >= MIN_PAIRS).sum()), 1)
        self.target_t64 = torch.from_numpy(self.target)
        self.target_ranks_t = torch_rank_rows(self.target)
        self._finite_t = torch.from_numpy(self.finite)
        self._target0_t = torch.where(self._finite_t, self.target_t64, torch.zeros((), dtype=torch.float64))
        n_lab = self._finite_t.sum(1)
        self._has_label = n_lab > 0
        self._ew_all = self._target0_t.sum(1) / n_lab.clamp(min=1).to(torch.float64)     # 基准:当天所有有标签的票等权

    # 分量 -> 适应度。单独拿出来是为了校准脚本和测试用同一个公式。
    def composite(self, comps: Dict[str, float]) -> float:
        used = [k for k, w in self.weights.items() if w > 0]
        if not all(np.isfinite(comps[k]) for k in used):
            return float("-inf")
        tot = sum(self.weights[k] for k in used)
        return float(self.unit * sum(self.weights[k] * comps[k] / self.scales[k] for k in used) / tot)

    def _side(self, x, ranks, order, n, book, half, sign: float) -> Dict[str, float]:
        """一个方向的四个分量。book / half 是排序位置上的掩码(哪些位置进书、哪些位置算多头端)。"""
        D, S = x.shape
        zero = torch.zeros((), dtype=torch.float64)
        k = book.sum(1).clamp(min=1).to(torch.float64).unsqueeze(1)
        w = torch.zeros((D, S), dtype=torch.float64).scatter_(1, order, torch.where(book, 1.0 / k, zero))
        gross = (w * self._target0_t).sum(1) - self._ew_all          # 标签缺的持仓在 _target0_t 里是 0
        traded = (w - torch.cat([torch.zeros((1, S), dtype=torch.float64), w[:-1]])).abs().sum(1)
        net = gross - self.cost_per_side * traded
        started = torch.cumsum((book.sum(1) > 0).to(torch.int64), 0) > 0
        live = started & self._has_label
        v = net[live]
        sharpe, ann = float("nan"), float("nan")
        sd = float(v.std(unbiased=False)) if v.numel() >= 3 else 0.0
        if sd > 1e-12:
            sharpe = float(v.mean() / sd * np.sqrt(252.0))
            ann = float(v.mean() * 252.0)
        sub = torch.zeros((D, S), dtype=torch.bool).scatter_(1, order, half)
        nan = torch.full((), float("nan"), dtype=torch.float64)
        ic = sign * nanmean_quiet(_cs_corr_days_t(torch.where(sub, x, nan), self.target_t64).numpy())
        ric = sign * nanmean_quiet(_cs_corr_days_t(torch.where(sub, ranks, nan), self.target_ranks_t).numpy())
        return {"long_sign": sign, "long_sharpe": sharpe, "long_ann": ann,
                "long_turnover": float(0.5 * traded.mean()), "long_ic": float(ic), "long_rank_ic": float(ric)}

    def _prepare(self, panel: np.ndarray, gated: bool):
        """排序一次,算老的三列和取值档位,过三道单因子门槛。返回 (out, 排序后的东西);被拒时后者是 None。"""
        cov = float((np.isfinite(np.asarray(panel)) & self.finite).sum() / self.cells)
        if gated and cov < self.min_coverage:           # 覆盖率不够的不用排序(一次排序是这里最贵的一步)
            nan = float("nan")
            return {"ic": nan, "rank_ic": nan, "turnover": nan, "fit": float("-inf"), "distinct": nan}, None
        x = torch.as_tensor(np.ascontiguousarray(panel)).to(torch.float64)
        ranks, order, n = _avg_ranks_t(x)
        fin = torch.isfinite(x)
        ric = nanmean_quiet(_cs_corr_days_t(ranks, self.target_ranks_t).numpy())
        ic_days = _cs_corr_days_t(x, self.target_t64).numpy()
        ic = nanmean_quiet(ic_days)
        out = {"ic": ic, "rank_ic": ric, "turnover": _book_turnover_t(order, n, self.top_frac), "fit": float("-inf")}
        # 每天不同取值的占比:排好序以后数相邻不相等的个数
        pos = torch.arange(x.shape[1]).unsqueeze(0)
        s = torch.gather(torch.where(fin, x, torch.full_like(x, float("inf"))), 1, order)
        inside = pos < n.unsqueeze(1)
        steps = ((s[:, 1:] != s[:, :-1]) & inside[:, 1:]).sum(1)
        enough = n >= MIN_PAIRS
        out["distinct"] = float(((steps[enough] + 1).to(torch.float64) / n[enough].to(torch.float64)).median()) \
            if bool(enough.any()) else 0.0
        scored = np.isfinite(ic_days).sum() / self.usable_days
        if gated and not (out["distinct"] >= self.min_distinct
                          and (self.min_scored_days <= 0 or scored >= self.min_scored_days)):
            return out, None
        return out, SimpleNamespace(x=x, ranks=ranks, order=order, n=n, s=s, pos=pos, inside=inside)

    @staticmethod
    def _book(p, sign: float, frac: float) -> torch.Tensor:
        """排序位置上的掩码:方向 +1 是因子值最高的 frac(和 long_short_weights 的多头腿同一批票),
        方向 -1 是最低的 frac(空头腿那批)。有值的票不到 2 只的日子不持仓。"""
        nn = p.n.unsqueeze(1)
        k = torch.clamp(torch.floor(p.n.to(torch.float64) * frac), min=1).to(torch.int64).unsqueeze(1)
        ok = nn >= 2
        return ok & (p.pos >= nn - k) & (p.pos < nn) if sign > 0 else ok & (p.pos < k)

    def _structure(self, p, book: torch.Tensor, sign: float) -> Tuple[float, float]:
        """(long_tie, long_code):书的边界是不是落在一段并列里;书附近的排名是不是就是股票列号。"""
        D, S = p.x.shape
        nn = p.n.unsqueeze(1)
        k = book.sum(1, keepdim=True).clamp(min=1)
        if sign > 0:
            near, edge = (nn >= 2) & (p.pos >= nn - 2 * k) & (p.pos < nn), (nn - k).clamp(min=0)
        else:
            near, edge = (nn >= 2) & (p.pos < 2 * k), (k - 1).clamp(min=0)
        # 和边界值相等、却在书外的票存在 -> 书里这段并列是靠代码顺序进来的
        tied = (p.s == torch.gather(p.s, 1, edge)) & p.inside
        crosses = (tied & ~book).any(1)
        share = torch.where(crosses, (tied & book).sum(1).to(torch.float64) / k.squeeze(1).to(torch.float64),
                            torch.zeros((), dtype=torch.float64))
        live_days = p.n >= 2
        tie = float(share[live_days].mean()) if bool(live_days.any()) else 1.0
        nan_t = torch.full((), float("nan"), dtype=torch.float64)
        posf, colf = p.pos.to(torch.float64).expand(D, S), p.order.to(torch.float64)
        code = nanmean_quiet(_cs_corr_days_t(torch.where(near, posf, nan_t), torch.where(near, colf, nan_t)).numpy())
        return tie, (float(code) if np.isfinite(code) else 0.0)

    def score(self, panel: np.ndarray, gated: bool = True, orientations=(1.0, -1.0)) -> Dict[str, float]:
        out, p = self._prepare(panel, gated)
        if p is None:
            return out
        best = None
        for sign in orientations:
            book, half = self._book(p, sign, self.top_frac), self._book(p, sign, self.side_frac)
            tie, code = self._structure(p, book, sign)
            side = self._side(p.x, p.ranks, p.order, p.n, book, half, float(sign))
            side["long_tie"], side["long_code"] = tie, code
            side["fit"] = self.composite({"sharpe": side["long_sharpe"], "ann": side["long_ann"],
                                          "ic": side["long_ic"], "rank_ic": side["long_rank_ic"]})
            if tie > self.max_boundary_ties or abs(code) > self.max_code_order:
                side["fit"] = float("-inf")
            if best is None or side["fit"] > best["fit"]:
                best = side
        out.update(best or {})
        return out

    def score_combined(self, panel: np.ndarray) -> Dict[str, float]:
        """池子的合成信号:方向是定的(往标签上拟合出来的),只做多它最高的那一头;不套覆盖率 / 取值档位 /
        可算天数这三道单因子门槛(并列和代码顺序那两道照套)。"""
        return self.score(panel, gated=False, orientations=(1.0,))


# --- 纯多头口径,第二版:命中率 + 几种做多组合 ------------------------------------------

HIT_TOPS = (0.05, 0.1, 0.2)  # "只做多头部"的几档
HIT_TAU = 0.1                # 指数加权命中率的衰减:离最头部每远 10% 的分位,权重乘 1/e
HIT_FLOOR = 3.0              # 合成时每个"分量 / 尺度"的下限(取负号),见 LongHitScorer
HIT_BOUNDARIES = {"z": "收益率 > 0", "m": "收益率 > 当天全市场等权收益"}
# 每个分量的尺度 = 现有因子库上它的"90 分位 - 中位数"(scripts/calibrate_long_hit.py,训练段前一半,
# 900 个随机抽的因子里过门槛的 684 个,2026-10-04),取两位有效数字。
HIT_SCALES: Dict[str, float] = {"sh_top5": 0.91, "sh_top10": 0.6, "sh_top20": 0.43, "sh_rankw": 0.2,
    "sh_ramp": 0.34, "xs_top5": 2.7, "xs_top10": 2.2, "xs_top20": 1.9, "xs_rankw": 2, "xs_ramp": 1.9,
    "xa_top5": 0.2, "xa_top10": 0.14, "xa_top20": 0.097, "xa_rankw": 0.045, "xa_ramp": 0.076, "hit_top5_z": 0.012,
    "hit_top10_z": 0.012, "hit_top20_z": 0.01, "hit_ramp_z": 0.0077, "hit_exp_z": 0.01, "auc_z": 0.0054,
    "auc_long_z": 0.0041, "hit_top5_m": 0.015, "hit_top10_m": 0.011, "hit_top20_m": 0.008, "hit_ramp_m": 0.0057,
    "hit_exp_m": 0.0088, "auc_m": 0.0028, "auc_long_m": 0.003, "lic": 0.0055, "lric": 0.014}
# 合成用哪几个、各占多少。同一份校准:前一半的分量去排,看后一半这个因子做多头的扣费超额夏普。
#   组合  太窄的书当奖励选不准(按前一半 top5 / top10 的超额夏普排在前 10% 的因子,后一半夏普中位数是
#         -0.85 / +0.38),宽的稳(top20 +1.51,按分位加权全做多 +0.50 但对平滑后的结果最准)。所以取
#         "只做多头部 20%"和"全部做多、权重正比于分位"两种,各看超额夏普和组合自己的夏普。
#   命中  以收益率 > 0 为界的几种都行(对平滑后结果的 Spearman 0.58~0.69),多头那一半里的 auc_long 最好,
#         指数加权的 hit_exp 次之;以"跑赢全市场等权"为界的那一组只有 0.2 左右,不用。命中率不含费用,
#         单独用会把高换手的也选进来(原样的结果只有 +0.2),要和组合一起用。
#   多头端 rank IC  单个分量里仍然是它最准(+0.78),留着,占三分之一。
# 三块各占三分之一。这组合成对后一半(半衰期 5 平滑、top10)的 Spearman 是 +0.78,第一版和老适应度都是 +0.68;
# 它排在前 5% 的因子后一半夏普中位数 +1.57(原样)/ +2.04(平滑),第一版 +1.56 / +1.99,老适应度 +0.36 / +1.22。
HIT_WEIGHTS: Dict[str, float] = {"xs_top20": 0.5, "xs_rankw": 0.5, "sh_top20": 0.5, "sh_rankw": 0.5,
                                 "auc_long_z": 1.0, "hit_exp_z": 1.0, "lric": 2.0}
_HIT_PORTS = tuple(f"top{int(round(f * 100))}" for f in HIT_TOPS) + ("rankw", "ramp")
_HIT_HITS = tuple(f"hit_top{int(round(f * 100))}" for f in HIT_TOPS) + ("hit_ramp", "hit_exp", "auc", "auc_long")
_HIT_COMPONENTS = tuple(f"{a}_{n}" for a in ("sh", "xs", "xa") for n in _HIT_PORTS) \
    + tuple(f"{h}_{b}" for b in HIT_BOUNDARIES for h in _HIT_HITS) + ("lic", "lric")


def _sharpe_ann(v: torch.Tensor) -> Tuple[float, float]:
    sd = float(v.std(unbiased=False)) if v.numel() >= 3 else 0.0
    if sd <= 1e-12:
        return float("nan"), float("nan")
    return float(v.mean() / sd * np.sqrt(252.0)), float(v.mean() * 252.0)


class LongHitScorer(LongScorer):
    """纯多头口径第二版(``--fitness long_hit``):多头命中率 + 几种用因子值做出来的纯多头组合。

    因子值先在截面上标准化成分位 u(0 = 最不看好,1 = 最看好;方向 -1 时把排序倒过来),下面的东西都只看 u。

    命中率("猜测命中率"):以收益率为正为界(``z``;另备一个以"跑赢当天全市场等权"为界的 ``m``),
    看收益为正的票落在因子排序的什么位置。几种算法都留着,合成用哪几个由校准定:
      hit_top5 / hit_top10 / hit_top20   头部 5% / 10% / 20% 里收益为正的占比 - 全市场当天收益为正的占比
      hit_ramp   加权命中率:权重从中位数处的 0 线性升到最头部的 1(只看多头那一半) - 全市场占比
      hit_exp    加权命中率:权重 exp(-(1-u)/tau),重心压在最头部一两成 - 全市场占比
      auc        全截面:两组 u 的均值差的一半(= (随便挑一只涨的、一只跌的,涨的排在前面的概率 - 0.5) / 2)
      auc_long   同上,只在多头那一半(u > 0.5)里比:两组 u 的均值差
    做多组合(权重和 1,每单位换手付 cost_per_side,基准是当天所有有标签的票等权):
      top5 / top10 / top20   只做多头部,等权
      rankw                  全部做多,权重正比于 u
      ramp                   只做多多头那一半,权重正比于 2u - 1
    每个组合记 sh_(扣费后组合自己的夏普 - 同一段日子里基准自己的夏普)、xs_(扣费后超额的夏普)、
    xa_(超额年化)、to_(单边日换手)。
    另外沿用第一版的多头端 IC / rank IC(lic / lric)。

    方向:两头都过得了并列 / 代码顺序门槛时,取"头部 10% 毛超额"高的那一头;其余只在选中的方向上算
    (第一版两个方向各算一整遍,这一版分量多,算两遍太慢)。合成信号只算 +1。
    适应度 = unit x 加权平均(分量 / 尺度),门槛和第一版是同一套。和第一版不同的一处:每个"分量 / 尺度"在
    -HIT_FLOOR 处截断 —— 超额夏普没有下界,而且扣费后为负的候选加一点跟踪误差夏普反而"变好",截断以后
    亏得厉害的候选不再靠噪声分高下,RL 拿到的奖励也有了下界。

    已知的偏向(2026-10-05 对抗复核量的,没有改):以"收益率 > 0"为界的命中率和多头端 rank IC 奖励的是中位数高,
    一个没有超额的纯低波动排序能拿到真因子相对噪声那段优势的 25%~40%;"全部做多按分位加权"减等权基准
    等价于一本按分位加权的多空书,只认得出差票的因子也能拿到约一半。读存档头部时要对着波动率排序看一眼。
    """

    KIND = "long_hit"
    MERGE_WEIGHTS = False
    PORTS, HITS, COMPONENTS = _HIT_PORTS, _HIT_HITS, _HIT_COMPONENTS
    DEFAULT_WEIGHTS = HIT_WEIGHTS
    DEFAULT_SCALES = HIT_SCALES

    def __init__(self, target: np.ndarray, *, boundaries: Optional[Sequence[str]] = None,
                 hit_tau: float = HIT_TAU, **kw) -> None:
        super().__init__(target, **kw)
        used = {k.rsplit("_", 1)[1] for k in self.weights if k.startswith(("hit_", "auc"))}
        self.boundaries = tuple(boundaries) if boundaries is not None else tuple(b for b in HIT_BOUNDARIES if b in used)
        if not set(self.boundaries) <= set(HIT_BOUNDARIES):
            raise ValueError(f"boundaries 只能取 {sorted(HIT_BOUNDARIES)}")
        if not used <= set(self.boundaries):
            raise ValueError(f"weights 里用到了分界 {sorted(used - set(self.boundaries))},boundaries 里却没有")
        self.hit_tau = float(hit_tau)
        zero = torch.zeros((), dtype=torch.float64)
        cut = {"z": torch.zeros((self.target.shape[0], 1), dtype=torch.float64), "m": self._ew_all.unsqueeze(1)}
        n_lab = self._finite_t.sum(1).clamp(min=1).to(torch.float64)
        self._win = {b: self._finite_t & (torch.where(self._finite_t, self.target_t64, zero) > cut[b])
                     for b in self.boundaries}
        self._base = {b: w.sum(1).to(torch.float64) / n_lab for b, w in self._win.items()}
        self.columns = ("long_sign",) + tuple(k for k, w in self.weights.items() if w > 0) \
            + ("to_top10", "long_tie", "long_code", "distinct")

    def composite(self, comps: Dict[str, float]) -> float:
        used = [k for k, w in self.weights.items() if w > 0]
        if not all(np.isfinite(comps[k]) for k in used):
            return float("-inf")
        tot = sum(self.weights[k] for k in used)
        return float(self.unit * sum(self.weights[k] * max(comps[k] / self.scales[k], -HIT_FLOOR) for k in used) / tot)

    def _gross_top(self, p, sign: float) -> float:
        """头部 top_frac 等权书的日均毛超额 —— 只用来在两个方向里挑一个。"""
        book = self._book(p, sign, self.top_frac)
        k = book.sum(1).clamp(min=1).to(torch.float64)
        tgt_s = torch.gather(self._target0_t, 1, p.order)
        g = torch.where(book, tgt_s, torch.zeros((), dtype=torch.float64)).sum(1) / k - self._ew_all
        live = (book.sum(1) > 0) & self._has_label
        return float(g[live].mean()) if bool(live.any()) else float("-inf")

    def _hit_side(self, p, sign: float) -> Dict[str, float]:
        """选定方向上的全部分量。都在排序位置上算,只有换手要把权重摆回股票原来的列。"""
        D, S = p.x.shape
        f64 = torch.float64
        zero = torch.zeros((), dtype=f64)
        ins = p.inside
        nf = p.n.to(f64).clamp(min=1).unsqueeze(1)
        u = (p.pos.to(f64) + 0.5) / nf                           # 排序位置的分位,(0, 1)
        if sign < 0:
            u = 1.0 - u
        u = torch.where(ins, u, zero)
        tgt_s = torch.gather(self._target0_t, 1, p.order)
        lab_s = torch.gather(self._finite_t, 1, p.order) & ins
        out: Dict[str, float] = {"long_sign": float(sign)}

        # --- 做多组合 ---
        ramp_w = torch.clamp(2.0 * u - 1.0, min=0.0)
        books = {f: self._book(p, sign, f) for f in HIT_TOPS}
        weights = {f"top{int(round(f * 100))}": torch.where(b, 1.0 / b.sum(1, keepdim=True).clamp(min=1).to(f64), zero)
                   for f, b in books.items()}
        weights["rankw"] = u / u.sum(1, keepdim=True).clamp(min=1e-300)
        weights["ramp"] = ramp_w / ramp_w.sum(1, keepdim=True).clamp(min=1e-300)
        first = torch.zeros((1, S), dtype=f64)
        for name, w_s in weights.items():
            w = torch.zeros((D, S), dtype=f64).scatter_(1, p.order, w_s)
            traded = (w - torch.cat([first, w[:-1]])).abs().sum(1)
            net = (w_s * tgt_s).sum(1) - self.cost_per_side * traded     # 标签缺的持仓在 tgt_s 里是 0
            live = (torch.cumsum((w_s.sum(1) > 0).to(torch.int64), 0) > 0) & self._has_label
            # 组合自己的夏普减去**同一段日子里**基准自己的夏普:不减的话每个候选都白带一份市场的夏普
            # (适应度 0 不再是"没本事"的线),而且预热期长的因子起算日晚,拿到的是另一段行情的夏普
            out[f"sh_{name}"] = _sharpe_ann(net[live])[0] - _sharpe_ann(self._ew_all[live])[0]
            out[f"xs_{name}"], out[f"xa_{name}"] = _sharpe_ann((net - self._ew_all)[live])
            out[f"to_{name}"] = float(0.5 * traded.mean())

        # --- 命中率 ---
        exp_w = torch.where(ins, torch.exp(-(1.0 - u) / self.hit_tau), zero)
        upper = ins & (u > 0.5)
        enough = p.n >= MIN_PAIRS

        def mean_days(v: torch.Tensor, ok: torch.Tensor) -> float:
            ok = ok & enough & torch.isfinite(v)
            return float(v[ok].mean()) if bool(ok.any()) else float("nan")

        def frac(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
            return num / den.clamp(min=1e-300)

        for b in self.boundaries:
            win_s = torch.gather(self._win[b], 1, p.order) & ins
            lose_s = lab_s & ~win_s
            base, winf, labf, losef = self._base[b], win_s.to(f64), lab_s.to(f64), lose_s.to(f64)
            for f, book in books.items():
                den = (book & lab_s).sum(1)
                out[f"hit_top{int(round(f * 100))}_{b}"] = mean_days(frac((book & win_s).sum(1).to(f64), den.to(f64)) - base,
                                                                 den > 0)
            for name, w_s in (("hit_ramp", ramp_w), ("hit_exp", exp_w)):
                den = (w_s * labf).sum(1)
                out[f"{name}_{b}"] = mean_days(frac((w_s * winf).sum(1), den) - base, den > 0)
            nw, nl = winf.sum(1), losef.sum(1)
            out[f"auc_{b}"] = mean_days(0.5 * (frac((u * winf).sum(1), nw) - frac((u * losef).sum(1), nl)),
                                        (nw > 0) & (nl > 0))
            upf = upper.to(f64)
            nw, nl = (winf * upf).sum(1), (losef * upf).sum(1)
            out[f"auc_long_{b}"] = mean_days(frac((u * winf * upf).sum(1), nw) - frac((u * losef * upf).sum(1), nl),
                                             (nw > 0) & (nl > 0))

        # --- 多头端 IC / rank IC(和第一版同一个口径)---
        half = self._book(p, sign, self.side_frac)
        sub = torch.zeros((D, S), dtype=torch.bool).scatter_(1, p.order, half)
        nan = torch.full((), float("nan"), dtype=f64)
        out["lic"] = float(sign * nanmean_quiet(_cs_corr_days_t(torch.where(sub, p.x, nan), self.target_t64).numpy()))
        out["lric"] = float(sign * nanmean_quiet(
            _cs_corr_days_t(torch.where(sub, p.ranks, nan), self.target_ranks_t).numpy()))
        return out

    def score(self, panel: np.ndarray, gated: bool = True, orientations=(1.0, -1.0)) -> Dict[str, float]:
        out, p = self._prepare(panel, gated)
        if p is None:
            return out
        best = None
        for sign in orientations:
            tie, code = self._structure(p, self._book(p, sign, self.top_frac), sign)
            if tie > self.max_boundary_ties or abs(code) > self.max_code_order:
                if best is None:
                    out.update(long_sign=float(sign), long_tie=tie, long_code=code)
                continue
            g = self._gross_top(p, sign)
            if best is None or g > best[0]:
                best = (g, float(sign), tie, code)
        if best is None:
            return out
        side = self._hit_side(p, best[1])
        side["long_tie"], side["long_code"] = best[2], best[3]
        side["fit"] = self.composite(side)
        out.update(side)
        return out


# --- 工厂注册表:给插件(比如 Partura 的 ParturaScorer)加新的 --fitness 取值 -----------------

SCORER_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register_scorer(kind: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """注册一个新的 ``--fitness`` 取值。``factory`` 会被 `make_scorer` 这样调用:

        factory(target, dates=<view.dates 或 None>, warmup=<dates 里的预热行数>,
                turnover_penalty=..., signed=..., cost_per_side=..., top_frac=...,
                min_coverage=..., min_scored_days=...)

    返回值必须满足打分器协议 —— 不是新发明的,是现有代码已经在按鸭子类型用的那四样:
        .score(panel, gated=True) -> dict,里面有 "fit"(有限数;或者 -inf,表示这个候选被拒)
        .score_combined(panel)    -> dict,里面有 "fit"                 (RL 的奖励路径用这个)
        .target_t64, .target_ranks_t                                   (GPMiner 两个都读)
        .columns  -> 存档里多出来的列名(可选)

    ``dates`` **包含**预热那些行,``dates[warmup]`` 才是第一个真正被打分的日子 —— factory 要
    靠这两个自己切 train / valid,`make_scorer` 本身不做任何切分,也不替内置的两个分支转发它们。

    ``kind`` 撞上 FITNESS_KINDS 里的内置名字一律报错(内置的不能被覆盖);撞上已经注册过的
    名字,除非 ``replace=True``,否则也报错。
    """
    if kind in FITNESS_KINDS:
        raise ValueError(f"{kind!r} is a built-in fitness kind, cannot be registered as a scorer factory")
    if kind in SCORER_FACTORIES and not replace:
        raise ValueError(f"scorer kind {kind!r} is already registered (pass replace=True to override it)")
    SCORER_FACTORIES[kind] = factory


def scorer_kinds() -> Tuple[str, ...]:
    """``--fitness`` 当前全部合法取值:四个内置的,加上插件注册过的(按名字排序)。"""
    return tuple(FITNESS_KINDS) + tuple(sorted(SCORER_FACTORIES))


def make_scorer(target: np.ndarray, kind: str, *, turnover_penalty: float = 0.0, signed: bool = False,
                cost_per_side: float = 0.0015, top_frac: float = BOOK_TOP_FRAC, min_coverage: float = 0.5,
                min_scored_days: float = 0.5, dates: Optional[Sequence[str]] = None, warmup: int = 0):
    """``kind`` -> 打分器。"long" / "long_hit" 是纯多头口径(换手按 cost_per_side 直接扣,turnover_penalty
    不用;后者加了命中率和几种做多组合),``dates``/``warmup`` 这两个内置分支都不转发;再查插件的
    SCORER_FACTORIES,查到就把 ``dates``/``warmup`` 连同其余参数一起转给它;两处都没有就退回老的
    ``|IC 或 rank IC| - turnover_penalty x 换手``(PanelScorer),未知 kind 在这里报同一个 ValueError。"""
    if kind in ("long", "long_hit"):
        cls = LongScorer if kind == "long" else LongHitScorer
        return cls(target, cost_per_side=cost_per_side, top_frac=top_frac, min_coverage=min_coverage,
                   min_scored_days=min_scored_days)
    factory = SCORER_FACTORIES.get(kind)
    if factory is not None:
        return factory(target, dates=dates, warmup=warmup, turnover_penalty=turnover_penalty, signed=signed,
                       cost_per_side=cost_per_side, top_frac=top_frac, min_coverage=min_coverage,
                       min_scored_days=min_scored_days)
    return PanelScorer(target, kind=kind, turnover_penalty=turnover_penalty, signed=signed, top_frac=top_frac,
                       min_coverage=min_coverage, min_scored_days=min_scored_days)


__all__ = ["PanelScorer", "LongScorer", "LongHitScorer", "make_scorer", "FITNESS_KINDS", "BOOK_TOP_FRAC",
           "LONG_COLUMNS", "LONG_SCALES", "LONG_WEIGHTS", "LONG_UNIT", "HIT_SCALES", "HIT_WEIGHTS", "HIT_TOPS",
           "HIT_FLOOR", "SCORER_FACTORIES", "register_scorer", "scorer_kinds"]
