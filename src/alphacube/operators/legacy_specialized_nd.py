"""Excluded, one-off / exotic / environment-specific ND functions.

These are NOT part of the generic AlphaFactorMining alpha-search operator
vocabulary (`nd_cs.py` / `nd_ts.py`) — none of them are tagged with the
`@op(...)` registry decorator from `operators.common`, and none of them
belong in `OPERATOR_REGISTRY`. They are copied here **verbatim** from the
original `/Users/wxy/Desktop/code/ND_operators.py`, unmodified and
un-vectorized, purely so that nothing is silently destroyed during the
rewrite. Do not import from this file into the operator registry.

Why each one is excluded (see the one-line comment above each def too):

- `ts_tail_hellinger` / `generate_tail_specialized_distributions` /
  `hellinger_distance`: a heavy, exotic tail-risk research experiment
  (Beta/Gamma/Normal candidate-distribution search + Hellinger-distance
  scoring per rolling window). Defaults to `device='cuda'`, which alone
  makes it unusable as a general CPU-safe operator. Not a standard
  Alpha101-style operator.
- `compute_ttm`: reads a HARDCODED absolute path
  (`/home/ubuntu/research/viewdb/features/by_time/{metric}/145900.feather`)
  that only exists on the original author's machine — this will raise
  `FileNotFoundError` (or worse, silently read someone else's unrelated
  file if a path happens to collide) on any other machine. It is also a
  full one-off financial-statement/TTM pipeline, not a generic tensor
  operator.
- `ts_growth` / `ts_denominator_k` / `ts_sparse_sum`: helpers for a
  specific sparse financial-announcement-data pipeline (values only
  present on announcement dates, everything else NaN), with heavy
  nested Python loops over `N` (and, inside each column, over the
  announcement events). Clearly a specific one-off pipeline, not a
  general rolling-window operator.

`merge_and_forward_fill` / `merge_and_forward_fill_vectorized` (used only
internally by `compute_ttm` in the original) were NOT carried over here —
per the rewrite scope, they're dropped along with `compute_ttm` itself.
`compute_ttm` below is kept only as a verbatim reference of the original
logic; since its `merge_and_forward_fill_vectorized` call has no
definition in this module, it is not actually callable as-is (it was
already unusable off the original machine anyway, due to the hardcoded
path above, so this doesn't reduce anything's functionality).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Exotic tail-risk experiment (hardcoded device='cuda', not a general op)
# ---------------------------------------------------------------------

def generate_tail_specialized_distributions(n, m, device='cuda'):
    # 1. Beta‐分布采样
    beta_a = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    beta_b = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    beta_dist = dist.Beta(beta_a, beta_b)
    tail_weights = beta_dist.sample((m, n)).squeeze(-1)  # (m, n)

    # 2. Gamma‐分布采样
    gamma_conc = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    gamma_rate = torch.ones(m, 1, device=device)
    gamma_dist = dist.Gamma(gamma_conc, gamma_rate)
    tail_strength = gamma_dist.sample((m, n)).squeeze(-1)  # (m, n)

    # 3. 正态分布采样
    normal_mean = torch.linspace(0.0, 1.0, m, device=device).unsqueeze(1)
    normal_std  = torch.linspace(0.1, 0.5, m, device=device).unsqueeze(1)
    normal_dist = dist.Normal(normal_mean, normal_std)
    middle_weights = normal_dist.sample((m, n)).squeeze(-1)  # (m, n)

    # 4. 组合 & 归一化
    combined = tail_weights * tail_strength + middle_weights   # (m, n)
    combined = torch.nn.functional.softmax(combined, dim=1)    # (m, n)

    # 5. 验证
    assert torch.allclose(
        combined.sum(dim=1),
        torch.ones(m, device=device),
        atol=1e-6
    ), "概率和不为1"

    return combined


def hellinger_distance(p, q):
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))


def ts_tail_hellinger(x: torch.Tensor,
                      window: int,
                      m:int = 5,
                      alpha: float = 0.2,
                      risk_free_rate: float = 0.0) -> torch.Tensor:
    """
    计算时间序列的尾部风险度量 TR，结合了初始 window 的逐点计算和后续的批量化计算。
    x: (T, N) 时间序列数据
    返回 TR: (T, N)，前 window 行会保持 NaN 或 0（因为样本不足）
    """
    T, N = x.shape
    device = x.device

    # 1) 生成候选分布
    D = generate_tail_specialized_distributions(window, m, device=device)  # (m, window)

    # 2) 预分配结果
    TR = torch.full((T, N), float('nan'), device=device)

    # 3) 处理前 window 时刻：逐点计算
    for t in range(window):
        if t == 0:
            # 没有历史，TR[0]=0（或保持 NaN）
            TR[0] = 0.0
            continue

        # 当前可用数据 (t, N)
        r_full = x[:t, :]

        # 过滤非有限值
        mask = torch.isfinite(r_full)
        r = torch.where(mask, r_full, torch.tensor(float('-inf'), device=device))

        # 排序 (t, N)
        sorted_r, _ = torch.sort(r, dim=0)

        # 计算经验分布 (t, N) via bucketize+scatter_add
        mins = sorted_r[0:1, :]            # (1, N)
        maxs = sorted_r[-1:, :]           # (1, N)
        edges = torch.linspace(0, 1, t+1, device=device)  # (t+1,)
        # 归一到 [0,1]
        normed = (sorted_r - mins) / (maxs - mins + 1e-8)

        # bucketize
        idx = torch.bucketize(normed, edges) - 1  # (t, N)

        # 修正 scatter_add 的维度问题
        emp = torch.zeros((t, N, t), device=device)  # (t, N, t)
        ones = torch.ones_like(idx, dtype=torch.float)

        # 确保 idx 和 ones 的维度正确
        idx_expanded = idx.permute(1, 0).unsqueeze(2)  # (N, t, 1)
        ones_expanded = ones.permute(1, 0).unsqueeze(2)  # (N, t, 1)

        emp = emp.scatter_add_(2, idx_expanded, ones_expanded)
        emp = emp.permute(0, 2, 1)  # (t, t, N)
        emp = emp / (emp.sum(dim=0, keepdim=True) + 1e-8)  # (t, t, N)

        # 批量计算 Hellinger + 定价误差
        # D[:, :t] → (m, t) → (1, m, t, 1)
        D4 = D[:, :t].view(1, m, t, 1)
        E4 = emp.unsqueeze(0)                          # (1, t, t, N)
        R4 = sorted_r.unsqueeze(1)                     # (t, 1, N)
        R4 = R4.permute(1, 0, 2).unsqueeze(-1)          # (1, t, N, 1) → broadcast to (1, m, t, N)

        # Hellinger & pricing
        h = torch.sqrt(0.5 * ((torch.sqrt(D4) - torch.sqrt(E4))**2).sum(dim=2))  # (1, m, N)
        eex = (D4 * R4).sum(dim=2)                                             # (1, m, N)
        p_err = torch.abs(eex - (1 + risk_free_rate))
        score = h + p_err                                                      # (1, m, N)

        # 最佳分布索引 & 取 dist
        best_i = score.argmin(dim=1).squeeze(0)                                 # (N,)
        onehot = F.one_hot(best_i, num_classes=m).float().unsqueeze(1)          # (N,1,m)
        D_sel = (D.unsqueeze(-1) * onehot).sum(dim=0)                           # (window, N) but we only need first t
        D_sel = D_sel[:t, :]

        # VaR idx & ES
        cdf = D_sel.cumsum(dim=0)                                               # (t, N)
        mask_cdf = cdf >= alpha
        var_idx = mask_cdf.float().argmax(dim=0)                                # (N,)
        var_vals = sorted_r[var_idx, torch.arange(N, device=device)]           # (N,)
        tail_mask = sorted_r <= var_vals.unsqueeze(0)
        pi = D_sel * tail_mask.float()
        TR[t] = (pi * (var_vals.unsqueeze(0) - sorted_r)).sum(dim=0)

    # 4) 处理 t ≥ window 的情况：批量化计算
    if T > window:
        # 用 unfold 一次性取出所有长度为 window 的滑动窗口
        W = x.unfold(0, window, 1)       # (T-window+1, window, N)

        # 过滤 NaN/Inf
        W = torch.where(torch.isfinite(W), W, torch.tensor(0., device=device))

        # 排序
        sorted_W, _ = torch.sort(W, dim=1)

        # 构造 histogram binning
        mins = sorted_W[:, 0, :].min(dim=1, keepdim=True)[0]     # (B, 1)
        maxs = sorted_W[:, -1, :].max(dim=1, keepdim=True)[0]    # (B, 1)
        edges = torch.linspace(0, 1, window+1, device=device)    # 相对位置 (window+1,)
        # 归一化到 [0,1]
        normed = (sorted_W - mins.unsqueeze(1)) / (maxs - mins + 1e-8).unsqueeze(1)
        normed = normed.clamp(0, 1)

        # bucketize
        idx = torch.bucketize(normed, edges) - 1  # (B, window, N)

        # 计算经验分布
        B = T - window + 1
        emp = torch.zeros((B, N, window), device=device)  # (B, N, window)
        ones = torch.ones_like(idx, dtype=torch.float)

        # 确保 idx 和 ones 的维度正确
        idx_expanded = idx.permute(0, 2, 1).unsqueeze(3)  # (B, N, window, 1)
        ones_expanded = ones.permute(0, 2, 1).unsqueeze(3)  # (B, N, window, 1)

        emp = emp.scatter_add_(2, idx_expanded, ones_expanded)
        emp = emp.permute(0, 2, 1)  # (B, window, N)
        emp = emp / (emp.sum(dim=1, keepdim=True) + 1e-8)

        # 计算 Hellinger 距离和定价误差
        D4 = D.view(1, m, window, 1)
        E4 = emp.unsqueeze(1)
        R4 = sorted_W.unsqueeze(1)

        h = torch.sqrt(0.5 * ((torch.sqrt(D4) - torch.sqrt(E4))**2).sum(dim=2))  # (B, m, N)
        eex = (D4 * R4).sum(dim=2)                          # (B, m, N)
        p_err = torch.abs(eex - (1 + risk_free_rate))      # (B, m, N)
        score = h + p_err                                  # (B, m, N)

        # 找到最优分布的索引
        best_i = score.argmin(dim=1)                       # (B, N)

        # 按 best_i 选出对应的 D 和 sorted_W
        D_b = D4.expand(B, m, window, N)
        onehot = F.one_hot(best_i, num_classes=m).permute(0, 2, 1).float()
        best_dist = (D_b * onehot.unsqueeze(2)).sum(dim=1)

        # 计算 VaR 和 ES
        cdf = best_dist.cumsum(dim=1)                      # (B, window, N)
        mask = cdf >= alpha
        var_idx = mask.float().argmax(dim=1)               # (B, N)

        var_vals = sorted_W[torch.arange(B, device=device).unsqueeze(1),
                           var_idx,                             # (B, N)
                           torch.arange(N, device=device).unsqueeze(0)]

        tail_mask = sorted_W <= var_vals.unsqueeze(1)
        pi = best_dist * tail_mask.float()
        diff = var_vals.unsqueeze(1) - sorted_W
        TR_batch = (pi * diff).sum(dim=1)                  # (B, N)

        # 填充结果
        TR[window:] = TR_batch

    return TR


# ---------------------------------------------------------------------
# TTM (trailing-twelve-months) financial-statement pipeline: hardcoded
# absolute path to the original author's machine, not portable.
# ---------------------------------------------------------------------

def compute_ttm(
    financial_tensors: List[torch.Tensor],
    metric: str,
    max_reports: int = 25,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    计算财务指标的 TTM 值，自动从 DataFrame 提取 statDate 和时间索引。

    Args:
        financial_tensors: [财报, Q4预, Q3预, Q2预, Q1预]，每个 shape (T, N)
        df: 原始数据框，含 datetime, symbol, {metric}_statDate 等列
        metric: 财务指标名，如 'revenue'
        max_reports: 每只股票最多财报数
        device: 计算设备

    Returns:
        TTM tensor, shape (T, N)
    """
    if len(financial_tensors) == 0:
        raise ValueError("financial_tensors 不能为空")
    if device is None:
        device = financial_tensors[0].device

    T, N = financial_tensors[0].shape

    # -------------------------------
    # Step 1: 从 df 构建 report_date_tensor 和 time_index
    # -------------------------------


    # 1. 原始 df
    df = pd.read_feather(f'/home/ubuntu/research/viewdb/features/by_time/{metric}/145900.feather')

    date_col = f"{metric}_statDate"
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    # 2. 提取子集后的 df_sub
    df_sub = df[['datetime', 'symbol', date_col]].copy()
    df_sub['date'] = pd.to_datetime(df_sub['datetime']).dt.date
    df_sub = df_sub.drop(columns=['datetime'])

    # 3. pivot 之后 df_pivot
    df_pivot = df_sub.pivot_table(
        index='date',
        columns='symbol',
        values=date_col,
        aggfunc='first',
        dropna=False
    )

    # 4. 排序之后 df_pivot
    df_pivot = df_pivot.sort_index()           # 按日期升序
    df_pivot = df_pivot.sort_index(axis=1)     # 按股票名升序

    # 5. 转换为 list 之后 report_date_tensor
    report_date_tensor = df_pivot.astype(str).replace('nan', np.nan).values.tolist()
    # 注意: 这里 report_date_tensor 是一个列表的列表，没有 .shape 属性
    # 所以我们打印其维度
    num_rows = len(report_date_tensor)
    num_cols = len(report_date_tensor[0]) if num_rows > 0 else 0

    # 6. final check for T and N
    T_income, N_income = financial_tensors[0].shape

    # 这里是之前你提供的代码，请保留
    for i in range(len(report_date_tensor)):
        for j in range(len(report_date_tensor[i])):
            if pd.isna(report_date_tensor[i][j]):
                report_date_tensor[i][j] = 'nan'

    time_index = pd.DatetimeIndex(df_pivot.index)

    # -------------------------------
    # Step 2: merge_and_forward_fill 得到 merged
    # -------------------------------
    # NOTE: `merge_and_forward_fill_vectorized` was intentionally dropped
    # (see module docstring) — this call is unresolved in this module.
    # compute_ttm is kept purely as a verbatim reference of the original
    # logic, not as a runnable function.
    merged = merge_and_forward_fill_vectorized(financial_tensors).to(device)

    # -------------------------------
    # Step 3: 自动推断 guidance_to_quarter
    # -------------------------------
    K = len(financial_tensors)
    guidance_to_quarter = [None]  # index 0 is financial report
    for i in range(1, min(K, 5)):
        guidance_to_quarter.append(f'Q{i}')
    while len(guidance_to_quarter) < K:
        guidance_to_quarter.append(None)

    # -------------------------------
    # Step 4: 提取每个股票的财报事件流
    # -------------------------------
    all_values = torch.full((N, max_reports), float('nan'), device=device)
    all_stat_dates = torch.zeros((N, max_reports), dtype=torch.int64, device=device)
    all_publish_times = torch.full((N, max_reports), -1, dtype=torch.long, device=device)

    income_data = financial_tensors[0].cpu().numpy()
    for n in range(N):
        values, sdates, pub_times = [], [], []
        for t in range(T):
            if pd.isna(income_data[t, n]) or report_date_tensor[t][n] in ('nan', '', 'None'):
                continue
            try:
                dt = pd.Timestamp(report_date_tensor[t][n])
                year, month, day = dt.year, dt.month, dt.day
                sdate_int = year * 10000 + month * 100 + day
                values.append(income_data[t, n])
                sdates.append(sdate_int)
                pub_times.append(t)
            except:
                continue
        L = min(len(values), max_reports)
        if L > 0:
            all_values[n, :L] = torch.tensor(values[:L], device=device, dtype=merged.dtype)
            all_stat_dates[n, :L] = torch.tensor(sdates[:L], device=device)
            all_publish_times[n, :L] = torch.tensor(pub_times[:L], device=device)

    # -------------------------------
    # Step 5: 构造 (T, N, max_reports) 的发布掩码和排序
    # -------------------------------
    t_idx = torch.arange(T, device=device).unsqueeze(1).unsqueeze(2)  # (T,1,1)
    pub_mask = all_publish_times.unsqueeze(0) <= t_idx  # (T, N, max_reports)

    values_exp = all_values.unsqueeze(0).expand(T, N, max_reports)
    dates_exp = all_stat_dates.unsqueeze(0).expand(T, N, max_reports)

    valid = pub_mask & ~torch.isnan(values_exp)

    # 按 stat_date 降序排序
    _, indices = torch.sort(dates_exp.where(valid, -1), dim=2, descending=True)
    sorted_values = torch.gather(values_exp, 2, indices)
    sorted_valid = torch.gather(valid, 2, indices)

    # 取前 4 个
    top4_mask = sorted_valid[:, :, :4]
    top4_values = sorted_values[:, :, :4]

    # A = 最近 4 个财报之和
    A_sum = top4_values.sum(dim=2)
    A_count = top4_mask.sum(dim=2)
    A = torch.where(A_count >= 4, A_sum, torch.where(A_count > 0, A_sum, torch.nan))

    # B = 最老的那个（第4个）
    B = torch.full_like(A, float('nan'))
    B = torch.where(A_count >= 4, top4_values[:, :, 3], B)

    # -------------------------------
    # Step 6: 构造 S = 是否来自预告
    # -------------------------------
    S = torch.zeros((T, N), device=device)
    income_report = financial_tensors[0].to(device)
    for i in range(1, K):
        tensor = financial_tensors[i].to(device)
        has_value = ~torch.isnan(tensor)
        no_report = torch.isnan(income_report)
        comes_from_guidance = has_value & no_report
        S = torch.where(comes_from_guidance, torch.tensor(1.0, device=device), S)

    # -------------------------------
    # Step 7: 应用公式 TTM = A + S * (merged - B)
    # -------------------------------
    TTM = A + S * (merged - B)

    return TTM


# ---------------------------------------------------------------------
# Sparse financial-announcement-data pipeline helpers: heavy nested
# Python loops, one-off pipeline (not a general rolling-window operator).
# ---------------------------------------------------------------------

def ts_growth(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    计算稀疏时间序列的 "k期前同比" 增长率
    即：当前公告值 vs 第 (k+1) 近的公告值（往前数第 k 个）

    输入: (T, N) 稀疏 tensor，每列一个时间序列
    输出: (T, N) 同比增长率，填充到下一个公告日前
    """
    T, N = x.shape
    device = x.device
    out = torch.full((T, N), torch.nan, dtype=x.dtype, device=device)

    for n in range(N):
        col = x[:, n]  # (T,)
        valid = ~torch.isnan(col)
        ann_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # (K,)
        K = len(ann_idx)

        if K == 0:
            continue

        # 可计算同比的起始位置：i >= k
        if K <= k:
            # 不足以计算同比，可选择填充 0 或保持 NaN
            out[:ann_idx[0], n] = 0.0  # 公告前
            for i in range(K):
                start = ann_idx[i]
                end = ann_idx[i+1] if i+1 < K else T
                out[start:end, n] = 0.0  # 无法同比，设为 0
            continue

        # 提取公告值
        ann_vals = col[ann_idx]  # (K,)

        # 计算同比增长率：从第 k 个公告开始
        # growth[i] = (ann_vals[i] - ann_vals[i-k]) / ann_vals[i-k]
        prev_vals = ann_vals[:-k]        # [v0, v1, ..., v_{K-k-1}]
        curr_vals = ann_vals[k:]         # [vk, v_{k+1}, ..., v_{K-1}]
        with torch.no_grad():
            growth = torch.where(
                prev_vals.abs() > 1e-8,
                (curr_vals - prev_vals) / prev_vals,
                torch.tensor(torch.nan, device=device)
            )  # (K - k,)

        # 填充区间：
        #   从 ann_idx[k]   开始，使用 growth[0] → 到 ann_idx[k+1]-1
        #   从 ann_idx[k+1] 开始，使用 growth[1] → 到 ann_idx[k+2]-1
        #   ...
        start_idx = ann_idx[k:]          # (K - k,)
        end_idx = torch.cat([
            ann_idx[k+1:],
            torch.tensor([T], device=device)
        ])  # (K - k,)

        # 向量化填充
        for i in range(len(start_idx)):
            out[start_idx[i]:end_idx[i], n] = growth[i]

        # 处理前 k 个公告日：无法计算同比，设为 0 或 NaN
        # 这里设为 0
        out[:ann_idx[0], n] = 0.0
        for i in range(min(k, K)):
            start = ann_idx[i]
            end = ann_idx[i+1] if i+1 < K else T
            out[start:end, n] = 0.0

    return out


def ts_sparse_sum(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    T x N tensor，每列独立处理：
    - 保持 NaN
    - 非 NaN 位置：若它是该列第 i 个非 NaN（i >= 1），且 i >= k，则输出前 k 个（含自己）的和；否则 NaN
    """
    T, N = tensor.shape
    out = torch.full((T, N), float('nan'), dtype=tensor.dtype, device=tensor.device)

    for col in range(N):
        col_data = tensor[:, col]                    # 提取列
        mask = ~torch.isnan(col_data)                # 找非 NaN
        indices = torch.nonzero(mask, as_tuple=True)[0]
        n = indices.size(0)
        if n < k:
            continue  # 不足 k 个，跳过

        vals = col_data[indices]                     # 提取非 NaN 值
        cumsum = torch.cumsum(vals, dim=0)           # 累积和
        # 计算滑动窗口和：sum[i-k+1:i+1] = cumsum[i] - cumsum[i-k]
        window_sum = cumsum[k-1:] - torch.cat([
            torch.zeros(1, dtype=cumsum.dtype, device=cumsum.device),
            cumsum[:n - k]
        ])

        # 构造结果：前 k-1 个为 NaN，后面填 window_sum
        result_vals = torch.full((n,), float('nan'), dtype=tensor.dtype, device=tensor.device)
        result_vals[k-1:] = window_sum

        # 写回原位置
        out[indices, col] = result_vals

    return out


def ts_denominator_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    输出“同比增长”中使用的分母：前 k 个公告值

    输入:
        x: (T, N) 稀疏 tensor，每列一个时间序列，非公告日为 NaN
        k: int，往前数第 k 个公告作为分母（k=1: 上一期, k=4: 同比）

    输出:
        denominator: (T, N)，每个位置是“应除以的值”
                     在前 k 个公告周期内为 NaN
    """
    T, N = x.shape
    device = x.device
    out = torch.full((T, N), torch.nan, dtype=x.dtype, device=device)

    for n in range(N):
        col = x[:, n]  # (T,)
        valid = ~torch.isnan(col)
        ann_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # (K,)
        K = len(ann_idx)

        if K == 0:
            continue

        # 提取公告值
        ann_vals = col[ann_idx]  # (K,)

        # 如果公告数不足 k+1，则全部无法计算分母
        if K <= k:
            # 所有公告区间都填 NaN（已初始化）
            # 可选：前向填充 0 或保持 NaN
            continue

        denom_vals = ann_vals[:-k]  # (K - k,)  → 对应第 k, k+1, ..., K-1 个公告

        # 填充区间：从 ann_idx[k] 开始，到下一个公告日前
        start_idx = ann_idx[k:]  # (K - k,)
        end_idx = torch.cat([
            ann_idx[k+1:],
            torch.tensor([T], device=device)
        ])  # (K - k,)

        # 向量化填充
        for i in range(len(start_idx)):
            out[start_idx[i]:end_idx[i], n] = denom_vals[i]

        # 前 k 个公告区间：保持 NaN（已初始化）
        # 如果你希望前 k 个公告日前也填 NaN，无需操作

    return out
