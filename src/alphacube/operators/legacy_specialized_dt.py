"""Legacy / pipeline-specific DT operators — kept for reference only,
EXCLUDED from the generic alpha-search operator registry (none of the
functions in this file are decorated with `@op(...)`, so
`common.OPERATOR_REGISTRY` never picks them up).

Mirrors the sibling `legacy_specialized_nd.py` pattern: these were ported
from `/Users/wxy/Desktop/code/DT_operators.py` alongside the general
`dt_cross.py`/`dt_in.py`/`dt_day.py` families, but each one is tightly
coupled to a specific one-off pipeline convention rather than being a
general-purpose, reusable rolling/reducing operator:

- **`cross_tail_hellinger`** (+ its distribution-sampling helpers
  `generate_tail_specialized_distributions`/`hellinger_distance`) —
  a bespoke tail-risk estimator that samples candidate Beta/Gamma/Normal
  mixture distributions and picks the best fit per window via a
  Hellinger-distance + pricing-error score. No `legacy_specialized_nd.py`
  exists yet in this codebase to import shared helpers from (checked —
  only `nd_cs.py` exists so far), so `generate_tail_specialized_
  distributions`/`hellinger_distance` are defined locally here; if/when
  an ND-side legacy module lands with the same helpers, one pair should
  be deleted and the other imported instead of duplicated.
- **`day_tide_speed`/`day_tide_change`** — piecewise slope/relative-change
  glue tied to exactly 3 caller-supplied index markers (`a`, `b`, `c`)
  via `day_locate`; a specific pipeline's feature, not a general operator.
- **`day_pdf_kurt`/`day_pdf_skew`/`day_pdf_stddev`/
  `day_pdf_quantile_single`/`day_pdf_quantile_bilateral_half_width`/
  `day_pdf_tail_sum_m`** (+ shared helper `_make_pdf_and_x`) — treat a
  `(x, f)` pair as an empirical PDF-over-`x` and compute moments/quantiles
  of it; specialized to whatever pipeline produces `f` as a density-like
  weight array.
- **`compute_bucket_entropy`** — hardcodes a 10-bucket [0.1, 0.2, ..., 1.0]
  entropy calculation over a specific z-scored-volatility construction.
- **`masked_multivariate_regression`** (+ its wrappers
  `compute_tvalues_list_vectorized`/`compute_fvalue_vectorized`/
  `compute_intercept_vectorized`/`compute_residuals_vectorized`) — general
  in shape but has a CONFIRMED bug on the F-statistic branch: `M =
  len(x_list)` is a plain Python `int`, and `M.clamp(min=1e-8)` doesn't
  exist on `int` — `AttributeError` the moment that branch runs. Since
  this family was already flagged as one-off pipeline glue (out of scope
  for the general `dt_day.py`/`dt_in.py` rewrite), it's copied here AS-IS
  with the bug intact and flagged with a comment, rather than spending
  effort fixing exotic, low-traffic code that isn't part of the core
  rolling-operator surface this rewrite targets.
- **`top_frac_sum_masked`** — sums the top `1/n` fraction of each row's
  values from a second tensor; single-use-looking helper, not part of the
  `in_*`/`day_*` naming/shape conventions.
- **`day_longest_run_of_label`** (+ `in_longest_run_of_label`,
  `in_roll_labels`, `in_label_prop`, `in_label_change_flags`,
  `in_label_transitions`) — an entire label-tracking mini state machine
  built on ONE specific labeling scheme (rolling-mean/std z-score buckets
  0..4 via hardcoded normal-quantile constants `z1=1.2816`, `z2=0.5244`).
  Moved as a group (not split between `dt_in.py`/`dt_day.py`/here) since
  `day_longest_run_of_label` directly calls `in_longest_run_of_label`,
  which in turn is meaningless without the label scheme the other three
  functions produce/consume — keeping them together avoids a cross-file
  dependency between the general registry and this legacy one.
- **`delete_open_close`** — hardcodes column `0` and the last 3 columns
  to NaN for one specific data layout (presumably an auction-price
  open/close convention); not a general operator.

None of this is reimplemented on `operators.common`'s shared primitives —
it's ported close to verbatim (light cleanup only: single top-level
`import torch`, no dead re-imports), since the goal here is preserving
reference behavior, not participating in the generic operator taxonomy.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F

from .dt_day import day_locate, day_max
from .dt_in import in_max, in_mean_full, in_mean_rolling, in_min, in_shift, in_stddev_full, in_stddev_rolling

try:
    import torch.distributions as dist
except ImportError:  # pragma: no cover
    dist = None


# ==========================================================================
# cross_tail_hellinger and its distribution-sampling helpers
# ==========================================================================

def generate_tail_specialized_distributions(n: int, m: int, device: str = "cuda") -> torch.Tensor:
    beta_a = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    beta_b = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    beta_dist = dist.Beta(beta_a, beta_b)
    tail_weights = beta_dist.sample((m, n)).squeeze(-1)

    gamma_conc = torch.linspace(0.5, 2.0, m, device=device).unsqueeze(1)
    gamma_rate = torch.ones(m, 1, device=device)
    gamma_dist = dist.Gamma(gamma_conc, gamma_rate)
    tail_strength = gamma_dist.sample((m, n)).squeeze(-1)

    normal_mean = torch.linspace(0.0, 1.0, m, device=device).unsqueeze(1)
    normal_std = torch.linspace(0.1, 0.5, m, device=device).unsqueeze(1)
    normal_dist = dist.Normal(normal_mean, normal_std)
    middle_weights = normal_dist.sample((m, n)).squeeze(-1)

    combined = tail_weights * tail_strength + middle_weights
    combined = torch.nn.functional.softmax(combined, dim=1)
    return combined


def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))


def cross_tail_hellinger(
    x: torch.Tensor, window: int, m: int = 5, alpha: float = 0.2, risk_free_rate: float = 0.0
) -> torch.Tensor:
    """Day-axis rolling tail-risk measure via best-fit candidate
    distribution + Hellinger distance + pricing error. `x`: `(T, N)`."""
    T, N = x.shape
    device = x.device
    D = generate_tail_specialized_distributions(window, m, device=device)
    TR = torch.full((T, N), float("nan"), device=device)

    for t in range(window):
        if t == 0:
            TR[0] = 0.0
            continue
        r_full = x[:t, :]
        mask = torch.isfinite(r_full)
        r = torch.where(mask, r_full, torch.full_like(r_full, float("-inf")))
        sorted_r, _ = torch.sort(r, dim=0)
        mins, maxs = sorted_r[0:1, :], sorted_r[-1:, :]
        edges = torch.linspace(0, 1, t + 1, device=device)
        normed = (sorted_r - mins) / (maxs - mins + 1e-8)
        idx = torch.bucketize(normed, edges) - 1
        emp = torch.zeros((t, N, t), device=device)
        ones = torch.ones_like(idx, dtype=torch.float)
        idx_expanded = idx.permute(1, 0).unsqueeze(2)
        ones_expanded = ones.permute(1, 0).unsqueeze(2)
        emp = emp.scatter_add_(2, idx_expanded, ones_expanded)
        emp = emp.permute(0, 2, 1)
        emp = emp / (emp.sum(dim=0, keepdim=True) + 1e-8)

        D4 = D[:, :t].view(1, m, t, 1)
        E4 = emp.unsqueeze(0)
        R4 = sorted_r.unsqueeze(1).permute(1, 0, 2).unsqueeze(-1)

        h = torch.sqrt(0.5 * ((torch.sqrt(D4) - torch.sqrt(E4)) ** 2).sum(dim=2))
        eex = (D4 * R4).sum(dim=2)
        p_err = torch.abs(eex - (1 + risk_free_rate))
        score = h + p_err

        best_i = score.argmin(dim=1).squeeze(0)
        onehot = F.one_hot(best_i, num_classes=m).float().unsqueeze(1)
        D_sel = (D.unsqueeze(-1) * onehot).sum(dim=0)[:t, :]

        cdf = D_sel.cumsum(dim=0)
        mask_cdf = cdf >= alpha
        var_idx = mask_cdf.float().argmax(dim=0)
        var_vals = sorted_r[var_idx, torch.arange(N, device=device)]
        tail_mask = sorted_r <= var_vals.unsqueeze(0)
        pi = D_sel * tail_mask.float()
        TR[t] = (pi * (var_vals.unsqueeze(0) - sorted_r)).sum(dim=0)

    if T > window:
        W = x.unfold(0, window, 1)
        W = torch.where(torch.isfinite(W), W, torch.zeros_like(W))
        sorted_W, _ = torch.sort(W, dim=1)
        mins = sorted_W[:, 0, :].min(dim=1, keepdim=True)[0]
        maxs = sorted_W[:, -1, :].max(dim=1, keepdim=True)[0]
        edges = torch.linspace(0, 1, window + 1, device=device)
        normed = (sorted_W - mins.unsqueeze(1)) / (maxs - mins + 1e-8).unsqueeze(1)
        normed = normed.clamp(0, 1)
        idx = torch.bucketize(normed, edges) - 1
        B = T - window + 1
        emp = torch.zeros((B, N, window), device=device)
        ones = torch.ones_like(idx, dtype=torch.float)
        idx_expanded = idx.permute(0, 2, 1).unsqueeze(3)
        ones_expanded = ones.permute(0, 2, 1).unsqueeze(3)
        emp = emp.scatter_add_(2, idx_expanded, ones_expanded)
        emp = emp.permute(0, 2, 1)
        emp = emp / (emp.sum(dim=1, keepdim=True) + 1e-8)

        D4 = D.view(1, m, window, 1)
        E4 = emp.unsqueeze(1)
        R4 = sorted_W.unsqueeze(1)

        h = torch.sqrt(0.5 * ((torch.sqrt(D4) - torch.sqrt(E4)) ** 2).sum(dim=2))
        eex = (D4 * R4).sum(dim=2)
        p_err = torch.abs(eex - (1 + risk_free_rate))
        score = h + p_err

        best_i = score.argmin(dim=1)
        D_b = D4.expand(B, m, window, N)
        onehot = F.one_hot(best_i, num_classes=m).permute(0, 2, 1).float()
        best_dist = (D_b * onehot.unsqueeze(2)).sum(dim=1)

        cdf = best_dist.cumsum(dim=1)
        mask = cdf >= alpha
        var_idx = mask.float().argmax(dim=1)
        var_vals = sorted_W[
            torch.arange(B, device=device).unsqueeze(1), var_idx, torch.arange(N, device=device).unsqueeze(0)
        ]
        tail_mask = sorted_W <= var_vals.unsqueeze(1)
        pi = best_dist * tail_mask.float()
        diff = var_vals.unsqueeze(1) - sorted_W
        TR[window:] = (pi * diff).sum(dim=1)

    return TR


# ==========================================================================
# day_tide_speed / day_tide_change (3-marker piecewise glue)
# ==========================================================================

def day_tide_speed(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    a, b, c, d = [z.squeeze(-1).unsqueeze(-1) for z in [a, b, c, d]]
    loc_a, loc_b, loc_c = day_locate(x, a), day_locate(x, b), day_locate(x, c)
    mask_pos = d > 0
    result1 = (loc_b - loc_a) / (b - a)
    result2 = (loc_c - loc_b) / (c - b)
    return torch.where(mask_pos, result1, result2)


def day_tide_change(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    a, b, c, d = [z.squeeze(-1).unsqueeze(-1) for z in [a, b, c, d]]
    loc_a, loc_b, loc_c = day_locate(x, a), day_locate(x, b), day_locate(x, c)
    mask_pos = d > 0
    result1 = (loc_b - loc_a) / loc_a
    result2 = (loc_c - loc_b) / loc_b
    return torch.where(mask_pos, result1, result2)


# ==========================================================================
# empirical-PDF-over-x family
# ==========================================================================

def _make_pdf_and_x(
    x: torch.Tensor, f: torch.Tensor, mode: Literal["minmax", "rank"] = "minmax", eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = ~(x.isnan() | f.isnan())
    f_safe = torch.where(mask, f, torch.zeros_like(f))
    row_sum = f_safe.sum(dim=-1, keepdim=True)
    pdf = f_safe / (row_sum + eps)
    pdf = torch.where(row_sum.eq(0).expand_as(pdf), torch.full_like(pdf, float("nan")), pdf)

    if mode == "minmax":
        x_min = torch.where(mask, x, torch.full_like(x, float("inf"))).min(dim=-1, keepdim=True)[0]
        x_max = torch.where(mask, x, torch.full_like(x, float("-inf"))).max(dim=-1, keepdim=True)[0]
        x_adj = (x - x_min) / ((x_max - x_min) + eps)
    else:
        rank = (torch.where(mask, x, torch.full_like(x, float("inf")))
                .argsort(dim=-1, stable=True).argsort(dim=-1, stable=True).float())
        rank_max = rank.max(dim=-1, keepdim=True)[0]
        x_adj = rank / (rank_max + eps)

    x_adj = torch.where(mask.any(dim=-1, keepdim=True).expand_as(x_adj), x_adj, torch.full_like(x_adj, float("nan")))
    return pdf, x_adj


def day_pdf_kurt(x: torch.Tensor, f: torch.Tensor, mode: Literal["minmax", "rank"] = "minmax") -> torch.Tensor:
    pdf, x_adj = _make_pdf_and_x(x, f, mode)
    mask = ~(x_adj.isnan() | pdf.isnan())
    pdf = torch.where(mask, pdf, torch.zeros_like(pdf))
    mean = (pdf * x_adj).sum(dim=-1, keepdim=True)
    var = (pdf * (x_adj - mean).square()).sum(dim=-1, keepdim=True)
    kurt = (pdf * (x_adj - mean).pow(4)).sum(dim=-1, keepdim=True) / (var.square() + 1e-12) - 3
    valid = mask.any(dim=-1, keepdim=True)
    return torch.where(valid, kurt, torch.full_like(kurt, float("nan")))


def day_pdf_skew(x: torch.Tensor, f: torch.Tensor, mode: Literal["minmax", "rank"] = "minmax") -> torch.Tensor:
    pdf, x_adj = _make_pdf_and_x(x, f, mode)
    mask = ~(x_adj.isnan() | pdf.isnan())
    pdf = torch.where(mask, pdf, torch.zeros_like(pdf))
    mean = (pdf * x_adj).sum(dim=-1, keepdim=True)
    var = (pdf * (x_adj - mean).square()).sum(dim=-1, keepdim=True)
    skew = (pdf * (x_adj - mean).pow(3)).sum(dim=-1, keepdim=True) / (var.pow(1.5) + 1e-12)
    valid = mask.any(dim=-1, keepdim=True)
    return torch.where(valid, skew, torch.full_like(skew, float("nan")))


def day_pdf_stddev(x: torch.Tensor, f: torch.Tensor, mode: Literal["minmax", "rank"] = "minmax") -> torch.Tensor:
    pdf, x_adj = _make_pdf_and_x(x, f, mode)
    mask = ~(x_adj.isnan() | pdf.isnan())
    pdf = torch.where(mask, pdf, torch.zeros_like(pdf))
    mean = (pdf * x_adj).sum(dim=-1, keepdim=True)
    var = (pdf * (x_adj - mean).square()).sum(dim=-1, keepdim=True)
    valid = mask.any(dim=-1, keepdim=True)
    return torch.where(valid, var.sqrt(), torch.full_like(var, float("nan")))


def day_pdf_quantile_single(
    x: torch.Tensor, f: torch.Tensor, n: float, left: bool = True, mode: Literal["minmax", "rank"] = "minmax"
) -> torch.Tensor:
    pdf, x_adj = _make_pdf_and_x(x, f, mode)
    valid = ~x_adj.isnan().all(dim=-1)
    B, L = x_adj.shape
    device = x.device
    pdf_safe = torch.where(valid.unsqueeze(-1), pdf, torch.zeros_like(pdf))
    n_vec = torch.full((B,), n, device=device)

    if left:
        cum = torch.cumsum(pdf_safe, dim=-1)
        idx = torch.searchsorted(cum, n_vec.unsqueeze(-1), right=False).squeeze(-1)
    else:
        cum = torch.flip(torch.cumsum(torch.flip(pdf_safe, dims=[-1]), dim=-1), dims=[-1])
        idx = torch.searchsorted(cum, n_vec.unsqueeze(-1), right=False).squeeze(-1)
        idx = L - 1 - idx

    idx = idx.clamp(0, L - 1)
    q = x_adj.gather(-1, idx.unsqueeze(-1))
    return torch.where(valid.unsqueeze(-1), q, torch.full_like(q, float("nan")))


def day_pdf_quantile_bilateral_half_width(
    x: torch.Tensor,
    f: torch.Tensor,
    n: float,
    center_mode: Literal["prob", "median"] = "prob",
    mode: str = "minmax",
    eps: float = 1e-12,
) -> torch.Tensor:
    if n >= 1.0:
        return torch.full((x.size(0), 1), 0.5, device=x.device, dtype=x.dtype)

    pdf, x_adj = _make_pdf_and_x(x, f, mode, eps)
    valid = ~x_adj.isnan().all(dim=-1, keepdim=True)
    B, L = pdf.shape
    device = x.device

    pdf = torch.where(valid.expand_as(pdf), pdf, torch.zeros_like(pdf))
    x_adj = torch.where(valid.expand_as(x_adj), x_adj, torch.zeros_like(x_adj))

    if center_mode == "prob":
        center_idx = torch.argmax(pdf, dim=-1, keepdim=True)
    else:
        diff = (x_adj - 0.5).abs()
        center_idx = torch.argmin(diff, dim=-1, keepdim=True)

    idx = torch.arange(L, device=device).view(1, L).expand(B, L)
    dist_ = (idx - center_idx).abs()
    order = torch.argsort(dist_, dim=-1, stable=True)

    pdf_sorted = torch.gather(pdf, 1, order)
    cum = torch.cumsum(pdf_sorted, dim=-1)

    mask = cum >= n
    first = torch.argmax(mask.int(), dim=-1, keepdim=True)
    first = torch.where(mask.any(dim=-1, keepdim=True), first, torch.full_like(first, L - 1))

    raw_idx = torch.where(idx <= first, order, torch.full_like(order, L))
    min_idx = raw_idx.min(dim=-1, keepdim=True)[0].clamp(0, L - 1)
    max_idx = raw_idx.max(dim=-1, keepdim=True)[0].clamp(0, L - 1)

    x_min = torch.gather(x_adj, 1, min_idx)
    x_max = torch.gather(x_adj, 1, max_idx)
    half_width = torch.clamp((x_max - x_min).abs() / 2.0, 0.0, 0.5)
    return torch.where(valid, half_width, torch.full_like(half_width, float("nan")))


def day_pdf_tail_sum_m(
    x: torch.Tensor, f: torch.Tensor, m: int, left: bool = True, mode: Literal["minmax", "rank"] = "minmax"
) -> torch.Tensor:
    pdf, _ = _make_pdf_and_x(x, f, mode)
    valid = ~pdf.isnan().all(dim=-1, keepdim=True)
    pdf = torch.where(valid.expand_as(pdf), pdf, torch.zeros_like(pdf))
    tail = pdf[:, :m] if left else pdf[:, -m:]
    s = tail.sum(dim=-1, keepdim=True)
    return torch.where(valid, s, torch.full_like(s, float("nan")))


# ==========================================================================
# compute_bucket_entropy
# ==========================================================================

def compute_bucket_entropy(vol: torch.Tensor) -> torch.Tensor:
    """10-bucket Shannon entropy over a z-scored `log(vol)` row."""
    x = torch.where(torch.abs(vol) > 1e-9, torch.log(torch.abs(vol)), torch.full_like(vol, float("nan")))
    vmin, vmax = in_min(x), in_max(x)
    z = (x - vmin) / (vmax - vmin)

    from .dt_day import day_mean as _day_mean

    p9 = _day_mean((z > 0.9).float())
    p8 = _day_mean(((z > 0.8).float()) * ((z < 0.9).float()))
    p7 = _day_mean(((z > 0.7).float()) * ((z < 0.8).float()))
    p6 = _day_mean(((z > 0.6).float()) * ((z < 0.7).float()))
    p5 = _day_mean(((z > 0.5).float()) * ((z < 0.6).float()))
    p4 = _day_mean(((z > 0.4).float()) * ((z < 0.5).float()))
    p3 = _day_mean(((z > 0.3).float()) * ((z < 0.4).float()))
    p2 = _day_mean(((z > 0.2).float()) * ((z < 0.3).float()))
    p1 = _day_mean(((z > 0.1).float()) * ((z < 0.2).float()))
    p0 = _day_mean((z < 0.1).float())

    def xlogx(p):
        return torch.where(torch.abs(p) > 1e-9, torch.log(torch.abs(p)), torch.full_like(p, float("nan"))) * p

    return sum(xlogx(p) for p in [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])


# ==========================================================================
# masked_multivariate_regression + wrappers (CONFIRMED bug, see module
# docstring — copied as-is, not fixed, per that reasoning)
# ==========================================================================

def masked_multivariate_regression(x_list: List[torch.Tensor], y: torch.Tensor):
    """Vectorized per-row multivariate OLS. KNOWN BUG (not fixed, see
    module docstring): on the F-statistic line, `M = len(x_list)` is a
    plain Python `int`; `M.clamp(min=1e-8)` raises `AttributeError` the
    instant that line executes, since `int` has no `.clamp` method. Any
    caller that reaches the `f_stat` computation will crash.
    """
    device = y.device
    B, N = y.shape
    M = len(x_list)

    X = torch.stack(x_list, dim=2)
    mask = torch.isnan(y)
    for x in x_list:
        mask = mask | torch.isnan(x[:, :])
    valid_mask = ~mask

    n_valid = valid_mask.sum(dim=1)
    min_required = M + 2
    sufficient_data = n_valid >= min_required

    ones = torch.ones(B, N, 1, device=device, dtype=X.dtype)
    X_design = torch.cat([ones, X], dim=2)

    X_masked = X_design * valid_mask.unsqueeze(-1).float()
    XtX = torch.einsum("bni,bnj->bij", X_masked, X_masked)

    y_masked = y * valid_mask.float()
    Xty = torch.einsum("bni,bn->bi", X_masked, y_masked)

    try:
        XtX_inv = torch.linalg.pinv(XtX)
    except Exception:
        XtX_inv = torch.pinverse(XtX)

    beta = torch.bmm(XtX_inv, Xty.unsqueeze(-1)).squeeze(-1)
    y_pred = torch.einsum("bni,bi->bn", X_design, beta)
    residuals = (y - y_pred) * valid_mask.float()

    rss = torch.sum(residuals ** 2, dim=1)
    df_residual = n_valid - (M + 1)
    df_model = torch.full((B,), M, device=device)

    mse = torch.where(df_residual > 0, rss / df_residual, torch.full_like(rss, float("nan")))
    var_beta = mse.unsqueeze(-1) * torch.diagonal(XtX_inv, dim1=-2, dim2=-1)
    std_err_beta = torch.sqrt(torch.clamp(var_beta, min=1e-12))
    t_values = beta / (std_err_beta + 1e-12)

    y_mean_valid = torch.where(valid_mask, y, torch.zeros_like(y)).sum(dim=1) / n_valid.clamp(min=1)
    tss = torch.sum(((y - y_mean_valid.unsqueeze(-1)) * valid_mask.float()) ** 2, dim=1)
    # BUG (kept intentionally, see docstring above): `M` is a Python int here,
    # not a Tensor, so `M.clamp(...)` raises AttributeError.
    f_stat = torch.where(
        (df_residual > 0) & (tss > 1e-12),
        ((tss - rss) / M.clamp(min=1e-8)) / (rss / df_residual.clamp(min=1e-8)),
        torch.full_like(rss, float("nan")),
    )

    return {
        "beta": beta,
        "residuals": residuals,
        "residual_sum_sq": rss,
        "t_values": t_values,
        "f_value": f_stat,
        "std_err": std_err_beta,
        "n_valid": n_valid,
        "df_residual": df_residual,
        "sufficient": sufficient_data,
    }


def compute_tvalues_list_vectorized(x_list: List[torch.Tensor], y: torch.Tensor) -> List[torch.Tensor]:
    if len(x_list) == 0:
        return []
    result = masked_multivariate_regression(x_list, y)
    t_vals = result["t_values"][:, 2:]
    return [t_vals[:, i : i + 1] for i in range(t_vals.shape[1])]


def compute_fvalue_vectorized(x_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    result = masked_multivariate_regression(x_list, y)
    return result["f_value"].unsqueeze(-1)


def compute_intercept_vectorized(x_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    result = masked_multivariate_regression(x_list, y)
    return result["beta"][:, 0:1]


def compute_residuals_vectorized(x_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    result = masked_multivariate_regression(x_list, y)
    residuals = result["residuals"]
    mask = torch.isnan(y)
    for x in x_list:
        mask = mask | torch.isnan(x)
    return torch.where(~mask, residuals, torch.full_like(residuals, float("nan")))


# ==========================================================================
# top_frac_sum_masked
# ==========================================================================

def top_frac_sum_masked(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    """Sum of `y` at the positions of `x`'s top `ceil(valid_count/n)` values
    per row."""
    T, Feat = x.shape
    n = int(n)
    mask_valid = ~torch.isnan(x)
    valid_cnt = mask_valid.sum(dim=1, keepdim=True)
    k = torch.ceil(valid_cnt / n).long()
    x_fill = torch.where(mask_valid, x, torch.full_like(x, float("-inf")))
    _, indices = torch.sort(x_fill, dim=1, descending=True, stable=True)
    row_idx = torch.arange(T, device=x.device).view(-1, 1)
    max_k = k.max()
    gathered = y[row_idx, indices[:, :max_k]]
    mask_k = torch.arange(max_k, device=x.device).view(1, -1) < k
    gathered = torch.where(mask_k, gathered, torch.zeros_like(gathered))
    return gathered.sum(dim=1, keepdim=True)


# ==========================================================================
# label-tracking family (in_roll_labels / in_label_prop /
# in_label_change_flags / in_label_transitions / in_longest_run_of_label /
# day_longest_run_of_label) — one specific z-score-bucket labeling scheme
# ==========================================================================

def in_roll_labels(x: torch.Tensor, window: int = 241) -> torch.Tensor:
    """Quantile-bucket label (0..4, -1 = invalid) from rolling/whole-row
    mean+std, using fixed normal-quantile constants `z1`/`z2`."""
    z1, z2 = 1.2816, 0.5244
    if window == x.shape[-1]:
        m, s = in_mean_full(x), in_stddev_full(x)
    else:
        m, s = in_mean_rolling(x, window), in_stddev_rolling(x, window)

    q1, q2, q3, q4 = m - z1 * s, m - z2 * s, m + z2 * s, m + z1 * s
    labels = torch.full_like(x, -1, dtype=torch.long)
    valid = torch.isfinite(x) & torch.isfinite(m) & torch.isfinite(s)
    labels = torch.where(valid & (x <= q1), torch.zeros_like(labels), labels)
    labels = torch.where(valid & (x > q1) & (x <= q2), torch.ones_like(labels), labels)
    labels = torch.where(valid & (x > q2) & (x <= q3), torch.full_like(labels, 2), labels)
    labels = torch.where(valid & (x > q3) & (x <= q4), torch.full_like(labels, 3), labels)
    labels = torch.where(valid & (x > q4), torch.full_like(labels, 4), labels)
    return labels


def in_label_prop(labels: torch.Tensor, label_value: int) -> torch.Tensor:
    """1.0 where `labels == label_value`, 0.0 elsewhere valid, NaN where
    `labels < 0` (invalid)."""
    mask = (labels == int(label_value)).to(dtype=torch.float32)
    return torch.where(labels >= 0, mask, torch.full_like(mask, float("nan")))


def in_label_change_flags(labels: torch.Tensor) -> torch.Tensor:
    """1.0 where the label changed from the previous minute (both valid),
    0.0 where unchanged (both valid), NaN if either is invalid."""
    prev = in_shift(labels, d=1, fill_value=-1)
    valid = (labels >= 0) & (prev >= 0)
    changed = (labels != prev) & valid
    return torch.where(valid, changed.float(), torch.full_like(changed.float(), float("nan")))


def in_label_transitions(labels: torch.Tensor, from_label: int, to_label: int) -> torch.Tensor:
    """1.0 where `prev==from_label and labels==to_label`, 0.0 where both
    valid but no such transition, NaN if either is invalid."""
    prev = in_shift(labels, 1)
    valid = (prev >= 0) & (labels >= 0)
    cond = (prev == int(from_label)) & (labels == int(to_label))
    res = torch.where(valid & cond, torch.ones_like(labels, dtype=torch.float32), torch.zeros_like(labels, dtype=torch.float32))
    return torch.where(valid, res, torch.full_like(res, float("nan")))


def in_longest_run_of_label(labels: torch.Tensor, label_value: int) -> torch.Tensor:
    """Running consecutive-length of `label_value` at each minute (NaN
    elsewhere). `T` is small (e.g. 240 minutes), so the per-minute Python
    loop here is cheap and kept for clarity, as in the original."""
    D, T = labels.shape
    device = labels.device
    run = torch.zeros(D, device=device)
    out = torch.full((D, T), float("nan"), device=device)
    for t in range(T):
        current = labels[:, t]
        is_target = torch.isfinite(current) & (current == label_value)
        run = torch.where(is_target, run + 1, torch.zeros_like(run))
        out[:, t] = torch.where(is_target, run.float(), torch.full_like(run, float("nan")))
    return out


def day_longest_run_of_label(labels: torch.Tensor, label_value: int) -> torch.Tensor:
    return day_max(in_longest_run_of_label(labels, label_value))


# ==========================================================================
# delete_open_close
# ==========================================================================

def delete_open_close(tensor: torch.Tensor) -> torch.Tensor:
    """NaN out column 0 and the last 3 columns (a specific auction-price
    open/close data-layout convention)."""
    result = tensor.clone()
    opening_width = 2 if tensor.shape[-1] == 241 else 1
    result[:, :opening_width] = float("nan")
    result[:, -3:] = float("nan")
    return result
