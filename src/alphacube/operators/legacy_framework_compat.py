"""用户 legacy 框架层辅助函数的忠实移植(factor_set_1_26.py 依赖)。

这些函数在最初清理建库(`ND_operators.py`/`DT_operators.py` → 本库)时没有
被搬进来 —— 它们不是"算子"而是当年挖掘/回测框架里的辅助工具,但用户的历史
因子表达式直接调用它们,所以按**原始定义逐个移植**(来源:
`~/Desktop/code/ND_operators.pyx` 与 `~/Desktop/code/DT_operators.py.bak_original`,
每个函数的 docstring 标注了原始行号)。行为上与原始一致,除非 docstring 里
明确写了偏差和原因。

命名前缀决定归类(见 `core/ops_full_catalog`):`in_*`/`cross_*`/`day_*` 归
DT,`ts_*`/`cs_*` 归 ND,其余共享;`EXPLICIT_CATEGORIES` 覆盖无前缀但有明确
家族语义的名字。`_max`/`_min`/`_pow` 原名以下划线开头会被目录扫描跳过,这里
用公开名定义,`core/expr_parser.OPERATOR_NAME_ALIASES` 把原名映射过来。
"""

from __future__ import annotations

from typing import List

import torch

from .common import rolling_ols

EPS = 1e-8  # 原始文件自己的 EPS(注意与 common.py 的 1e-9 不同,保持原值)

# 无法从名字前缀推断家族的显式归类(被 ops_full_catalog 消费)。
EXPLICIT_CATEGORIES = {
    "column_correlation": "ts",   # 整列(全样本时间轴)统计 → ND
    "scale": "cs",                # 截面绝对值和归一 → ND cs 语义
}


# --------------------------------------------------------------------------
# 共享 elementwise(原 ND_operators.pyx)
# --------------------------------------------------------------------------

def create_valid_mask(x: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """原 ND_operators.pyx:30 —— 注意语义就是 ``x > threshold``(不是
    isfinite):legacy 用它把"权重/价格>0"当作有效域。"""
    return x > threshold


def create_ones(x: torch.Tensor, d=1) -> torch.Tensor:
    """原 DT_operators:1291 —— 与原始一致:`d` 参数被无视,恒返回全 1。"""
    return torch.ones_like(x)


def pairwise_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """原 ND_operators.pyx:92 `_max` —— 逐元素 max(NaN 传播同 torch)。"""
    return torch.maximum(x, y)


def pairwise_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """原 ND_operators.pyx:99 `_min`。"""
    return torch.minimum(x, y)


def signed_pow(x: torch.Tensor, e) -> torch.Tensor:
    """原 ND_operators.pyx:127 `_pow`:e<=1 用 |x|^e*sign(x)(保号防复数),
    e>1 用普通幂。"""
    if isinstance(e, (int, float)):
        e = torch.tensor(e, dtype=x.dtype, device=x.device)
    x, e = torch.broadcast_tensors(x, e)
    return torch.where(
        e <= 1.0,
        torch.mul(torch.pow(x.abs(), e), torch.sign(x)),
        torch.pow(x, e),
    )


def greater_equal(x: torch.Tensor, y) -> torch.Tensor:
    """`a >= b` 比较语法糖的落点(torch.ge:NaN 参与比较恒 False,与
    greater/less 的既有 NaN 语义一致)。"""
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y), dtype=x.dtype, device=x.device)
    return torch.ge(x, y).to(x.dtype)


def less_equal(x: torch.Tensor, y) -> torch.Tensor:
    """`a <= b` 比较语法糖的落点。"""
    if isinstance(y, (int, float)):
        y = torch.tensor(float(y), dtype=x.dtype, device=x.device)
    return torch.le(x, y).to(x.dtype)


def replace_small_with_nan(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """原 DT_operators:4745 —— |x|<eps → NaN。"""
    return torch.where(
        torch.abs(tensor) < eps,
        torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device),
        tensor,
    )


def tensor_list_std(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """原 DT_operators:5010 —— 对 list 内同形张量按位置求非 NaN 样本标准差
    (Bessel 校正,count<=1 处为 NaN)。"""
    if len(tensor_list) == 0:
        raise ValueError("tensor_list 不能为空")
    stacked = torch.stack(list(tensor_list), dim=0)
    mask = ~torch.isnan(stacked)
    filled = torch.where(mask, stacked, torch.zeros((), dtype=stacked.dtype, device=stacked.device))
    count = mask.sum(dim=0).to(stacked.dtype)
    total = filled.sum(dim=0)
    nan = torch.tensor(float("nan"), dtype=stacked.dtype, device=stacked.device)
    mean = torch.where(count > 0, total / count.clamp(min=1), nan)
    sq_sum = torch.where(mask, stacked**2, torch.zeros((), dtype=stacked.dtype, device=stacked.device)).sum(dim=0)
    var = torch.where(count > 0, sq_sum / count.clamp(min=1) - mean**2, nan)
    var_unbiased = torch.where(count > 1, var * count / (count - 1).clamp(min=1), var)
    std = torch.sqrt(torch.clamp(var_unbiased, min=0.0))
    return torch.where(count > 1, std, nan)


# --------------------------------------------------------------------------
# ND(原 ND_operators.pyx)
# --------------------------------------------------------------------------

def scale(x: torch.Tensor, a: float = 1.0) -> torch.Tensor:
    """原 ND_operators.pyx:742 —— WorldQuant 式截面归一:x / sum(|x|)。"""
    denom = torch.sum(torch.abs(x), dim=-1, keepdim=True)
    denom = torch.where(denom < EPS, torch.ones_like(denom), denom)
    return a * x / denom


def ts_all_mean(tensor: torch.Tensor) -> torch.Tensor:
    """原 ND_operators.pyx:1525 —— 每列全样本非 NaN 均值,广播回原形状。
    注意:这是全样本统计(含未来数据),是 legacy 因子自己的设计选择。"""
    not_nan = ~torch.isnan(tensor)
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    col_sum = torch.where(not_nan, tensor, zero).sum(dim=0)
    col_count = not_nan.sum(dim=0)
    nan = torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    col_mean = torch.where(col_count > 0, col_sum / col_count.clamp(min=1), nan)
    return col_mean.unsqueeze(0).expand_as(tensor)


def ts_column_nonnull_sum(x: torch.Tensor) -> torch.Tensor:
    """原 ND_operators.pyx:1950 —— 每列非 NaN 和,广播回原形状;原 NaN 位置
    保持 NaN。同为全样本统计,见 `ts_all_mean` 的注意。"""
    not_nan = ~torch.isnan(x)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    col_sum = torch.where(not_nan, x, zero).sum(dim=0, keepdim=True)
    out = col_sum.expand_as(x).clone()
    out[~not_nan] = float("nan")
    return out


def column_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """原 ND_operators.pyx:1891 —— 每列全样本相关系数(仅双方同时非 NaN 的
    行参与;有效点<2 的列为 NaN;常数列为 0)。

    与原始的唯一偏差:原始返回 (1, N),这里 expand 成与输入同形的 (T, N)
    (每列常数),使它能作为因子根/中间节点参与后续逐日运算 —— 数值内容
    完全一致,只是复制到了每一行。
    """
    mask = ~torch.isnan(x) & ~torch.isnan(y)
    valid_count = mask.sum(dim=0)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    x_f = torch.where(mask, x, zero)
    y_f = torch.where(mask, y, zero)
    cnt = valid_count.clamp(min=1).to(x.dtype)
    x_mean = x_f.sum(dim=0) / cnt
    y_mean = y_f.sum(dim=0) / cnt
    x_c = torch.where(mask, x - x_mean, zero)
    y_c = torch.where(mask, y - y_mean, zero)
    cov = (x_c * y_c).sum(dim=0)
    denom = torch.sqrt((x_c * x_c).sum(dim=0) * (y_c * y_c).sum(dim=0))
    corr = torch.where(denom > 1e-10, cov / denom, torch.zeros_like(cov))
    corr = torch.where(valid_count >= 2, corr,
                       torch.tensor(float("nan"), dtype=x.dtype, device=x.device))
    return corr.unsqueeze(0).expand_as(x)


# --------------------------------------------------------------------------
# 多张量池化滚动统计(list 操作数)
# 原 ND_operators.pyx:1797/1872/1881;DT 版本(3109/3442)与 ND 版本逐行相同,
# 只是滚动轴向:in_* 沿最后一维(分钟),ts_* 沿第一维(交易日,转置复用)。
# --------------------------------------------------------------------------

def in_mean_rolling_multi(x_list: List[torch.Tensor], d: int) -> torch.Tensor:
    """窗口内把 m 个张量的 m*d 个样本合并求 nanmean;前 d-1 列用递增窗口。"""
    m = len(x_list)
    B, N = x_list[0].shape
    d = min(int(d), N)
    if d <= 0:
        return torch.full((B, N), float("nan"), device=x_list[0].device)
    x_stack = torch.stack(list(x_list), dim=0)
    res = torch.full((B, N), float("nan"), device=x_stack.device, dtype=x_stack.dtype)
    if N >= d:
        unfolded = x_stack.unfold(2, d, 1)
        flat = unfolded.permute(1, 2, 0, 3).reshape(B, N - d + 1, m * d)
        res[:, d - 1:] = flat.nanmean(dim=-1)
    if d > 1:
        L = d - 1
        x_crop = x_stack[..., :L]
        expanded = x_crop.unsqueeze(-1).expand(m, B, L, L)
        lower = torch.tril(torch.ones(L, L, device=x_stack.device, dtype=torch.bool))
        windowed = torch.where(lower.permute(1, 0), expanded, torch.tensor(float("nan"), dtype=x_stack.dtype, device=x_stack.device))
        flat_win = windowed.permute(1, 3, 0, 2).reshape(B, L, m * L)
        res[:, :L] = flat_win.nanmean(dim=-1)
    return res


def in_stddev_rolling_multi(x_list: List[torch.Tensor], d: int, ddof: int = 1,
                            min_periods: int = 2) -> torch.Tensor:
    """同上,合并样本求样本标准差(有效样本数 < min_periods 为 NaN)。"""
    m = len(x_list)
    B, N = x_list[0].shape
    d = min(int(d), N)
    if d <= 0:
        return torch.full((B, N), float("nan"), device=x_list[0].device)
    x_stack = torch.stack(list(x_list), dim=0)
    res = torch.full((B, N), float("nan"), device=x_stack.device, dtype=x_stack.dtype)
    nan = torch.tensor(float("nan"), dtype=x_stack.dtype, device=x_stack.device)
    zero = torch.zeros((), dtype=x_stack.dtype, device=x_stack.device)
    if N >= d:
        unfolded = x_stack.unfold(2, d, 1)
        flat = unfolded.permute(1, 2, 0, 3).reshape(B, N - d + 1, m * d)
        mask = ~torch.isnan(flat)
        count = mask.sum(-1)
        mean = torch.where(mask, flat, zero).sum(-1) / count.clamp(min=1)
        var = torch.where(mask, (flat - mean.unsqueeze(-1)) ** 2, zero).sum(-1) / (
            count - ddof).clamp(min=1e-8)
        res[:, d - 1:] = torch.where(count >= min_periods, torch.sqrt(var), nan)
    if d > 1:
        L = d - 1
        x_crop = x_stack[..., :L]
        expanded = x_crop.unsqueeze(-1).expand(m, B, L, L)
        lower = torch.tril(torch.ones(L, L, device=x_stack.device, dtype=torch.bool))
        windowed = torch.where(lower.permute(1, 0), expanded, nan)
        flat_win = windowed.permute(1, 3, 0, 2).reshape(B, L, m * L)
        mask_w = ~torch.isnan(flat_win)
        count_w = mask_w.sum(-1)
        mean_w = torch.where(mask_w, flat_win, zero).sum(-1) / count_w.clamp(min=1)
        var_w = torch.where(mask_w, (flat_win - mean_w.unsqueeze(-1)) ** 2, zero).sum(-1) / (
            count_w - ddof).clamp(min=1e-8)
        res[:, :L] = torch.where(count_w >= min_periods, torch.sqrt(var_w), nan)
    return res


def ts_mean_rolling_multi(x_list: List[torch.Tensor], d: int) -> torch.Tensor:
    """行方向(时间轴)版本 —— 原始实现即转置复用列方向函数。"""
    return in_mean_rolling_multi([x.t() for x in x_list], d).t()


def ts_stddev_rolling_multi(x_list: List[torch.Tensor], d: int, ddof: int = 1,
                            min_periods: int = 2) -> torch.Tensor:
    return in_stddev_rolling_multi([x.t() for x in x_list], d, ddof, min_periods).t()


# --------------------------------------------------------------------------
# DT(原 DT_operators.py.bak_original)
# --------------------------------------------------------------------------

def day_quantile_efficient(x: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    """原 DT_operators:1866 —— torch.nanquantile 的行内分位数,(B, 1)。"""
    return torch.nanquantile(x, q, dim=1, keepdim=True)


def in_shift_left(tensor: torch.Tensor) -> torch.Tensor:
    """原 DT_operators:2115 —— 行内整体左移 1 位,第 0 列补 0。"""
    shifted = tensor[:, 1:]
    pad = torch.zeros(tensor.size(0), 1, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([pad, shifted], dim=1)


def in_shift_right(tensor: torch.Tensor) -> torch.Tensor:
    """原 DT_operators:2135 —— 行内整体右移 1 位,最后一列补 0。"""
    shifted = tensor[:, :-1]
    pad = torch.zeros(tensor.size(0), 1, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([shifted, pad], dim=1)


def day_skew_masked(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """原 DT_operators:2222 —— mask==1 位置的行内偏度,输出 (B, 1)。"""
    m = m.to(x.dtype)
    n = m.sum(dim=1, keepdim=True)
    mu = (x * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    var = ((x - mu) ** 2 * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    std = torch.sqrt(var.clamp(min=1e-8))
    m3 = ((x - mu) ** 3 * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    return m3 / (std**3)


def day_kurt_masked(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """原 DT_operators 同区 —— mask 内峰度(Fisher, 减 3),输出 (B, 1)。"""
    m = m.to(x.dtype)
    n = m.sum(dim=1, keepdim=True)
    mu = (x * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    var = ((x - mu) ** 2 * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    std = torch.sqrt(var.clamp(min=1e-8))
    m4 = ((x - mu) ** 4 * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    return m4 / (std**4) - 3.0


def in_multireg_legacy(x_list: List[torch.Tensor], y: torch.Tensor, d: int,
                       output: str = "beta", drop_first_t: bool = True):
    """原 DT_operators:3766 `in_multireg(x_list, y, d, output, drop_first_t)`
    的兼容签名(内置库的 `in_multireg` 是 3 参返回 dict 的新 API,通过
    `OPERATOR_NAME_ALIASES` 把老写法路由到这里)。

    **与原始的已知偏差(沿用本库对整个 multireg 家族的既有决策,见
    `dt_cross.py` 模块 docstring)**:原始走 `cross_multireg_core` 的联合多元
    回归(且那个 core 有已记录的 bug);本库把 multireg 家族统一简化为
    "每个 predictor 独立单变量滚动 OLS"。因此:
    - 'tvalue'/'beta' → 每个 predictor 一个面板的 list(tvalue 按原始的
      `drop_first_t` 约定去掉第一个);
    - 'alpha'/'fvalue' → 原始联合回归只有一个截距/一个 F 值,独立回归有
      m 个 —— 这里取 m 个的均值作为单一面板,保持"单张量输出"的原始接口;
    - 'resid' → y 减去每个 predictor 各自的 beta*x 贡献(与
      `in_multiregres` 相同定义)。
    """
    valid = {"alpha", "beta", "tvalue", "fvalue", "resid"}
    if output not in valid:
        raise ValueError(f"output must be one of {valid}, got {output}")
    d = int(d)
    results = [rolling_ols(x, y, d, axis=-1) for x in x_list]
    if output == "beta":
        return [r["beta"] for r in results]
    if output == "tvalue":
        tvals = [r["tvalue"] for r in results]
        if drop_first_t and len(tvals) > 1:
            return tvals[1:]
        return tvals
    if output == "alpha":
        return torch.stack([r["alpha"] for r in results], dim=0).nanmean(dim=0)
    if output == "fvalue":
        return torch.stack([r["fvalue"] for r in results], dim=0).nanmean(dim=0)
    res = y
    for x, r in zip(x_list, results):
        res = res - r["beta"] * x
    return res
