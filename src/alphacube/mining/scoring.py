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
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from ..metrics import MIN_PAIRS, _cs_corr_days_t, nanmean_quiet, rank_ic_turnover, torch_rank_rows

FITNESS_KINDS = ("ic", "rank_ic")
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
        if kind not in FITNESS_KINDS:
            raise ValueError(f"kind must be one of {FITNESS_KINDS}, not {kind!r}")
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


__all__ = ["PanelScorer", "FITNESS_KINDS", "BOOK_TOP_FRAC"]
