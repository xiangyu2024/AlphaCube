"""Spectral ("FFT energy-ratio") DT operators — take the FFT along `dim=-1`
(M, the intraday-minute axis) and reduce the resulting power spectrum down
to a single scalar per day, shape `(D, 1)`.

DT tensors are shape `(D, M)` = (day, intraday-minute). Every operator in
this file is row-independent (day `d`'s output depends only on day `d`'s
minute series) and DAY-COLLAPSING: all four public `energy_ratio_*`
functions return `(D, 1)`, never `(D, M)`.

Ported from the original `/Users/wxy/Desktop/code/DT_operators.py`
(lines ~3884-4030), the `fft_with_nan_handling` / `energy_ratio_*` family.
`dt_in.py`'s port note originally listed this family as "dropped ... a
candidate for its own `dt_spectral.py` later if needed"; this is that file.

## Fidelity contract: NOTHING here was "fixed"

Unlike `dt_in.py` / `dt_day.py` / `dt_cross.py`, which deliberately
corrected axis-convention and NaN-handling bugs while porting, this module
is a BIT-EXACT replay of the legacy implementation. These operators feed a
live research factor set whose historical factor values must reproduce
exactly, so every quirk below is preserved on purpose and merely
documented. `tests/test_dt_spectral.py` pins this by executing the legacy
source side-by-side and comparing raw bits.

Preserved legacy quirks (do NOT "clean these up"):

1. **`fft_with_nan_handling` does not compact valid samples.** It replaces
   NaN with `0.0` IN PLACE, then separately zeroes every column at index
   `>= valid_points` (the row's NaN count subtracted from its length). For
   a row whose NaNs sit in the MIDDLE, those two masks are not the same
   set: the interior NaN stays a zero-hole in the signal AND a tail of
   otherwise-valid samples gets zeroed out too. The transform is therefore
   of a zero-padded, partially-truncated series, not of the compacted
   valid samples. Only for trailing-NaN rows (the common intraday case)
   do the two coincide.

2. **`pad_to_pow2` truncates rather than only padding.** `max_len` is
   derived from the max valid count ACROSS ALL ROWS. When
   `pad_to_pow2=False` and some row is shorter than `L`, `max_len < L`
   and the branch takes `x_clean[:, :max_len]` — every row is CROPPED to
   the longest row's valid count, so the output spectrum width depends on
   the batch's NaN pattern, not on `M` alone.

3. **`energy_ratio_n`'s denominator switches meaning with `pad_to_pow2`.**
   With `pad_to_pow2=True` it divides the frequency-bin count `k` by
   `valid_points` (a count of TIME samples); with `False` it divides by
   `M` (the FFT length). The two branches are not the same normalization.

4. **`energy_ratio_band_n` mixes a bin index into a sample count.**
   `band_valid` is `valid_points - N//2` (high band) or
   `min(valid_points, (N+1)//2)` (low band) — subtracting/clamping a
   FREQUENCY-BIN offset against a TIME-SAMPLE count. Dimensionally
   inconsistent, but load-bearing for reproducing historical values.

5. **`energy_ratio_topm_minus_q`'s `assert m > q` fires before `m` is
   clamped.** `m = min(m, power_spectrum.shape[1])` happens AFTER the
   assert, so a call with `m > q` that gets clamped to `m <= q` slips
   past the guard and yields an empty `topm_vals[:, q:]` slice whose sum
   is `0.0` — a silent zero, not an error.

6. **`energy_ratio_topm` / `energy_ratio_topm_minus_q` fill all-NaN rows
   with `-inf` before `topk`.** The `-inf` values are summed into
   `topm_energy` for those rows, but the trailing `torch.where` overwrites
   the result with NaN anyway, so the `-inf` never escapes.

## Category tag: `"day"`, NOT `"in"` (measured, not assumed)

These operators consume the minute axis entirely — every one returns
`(D, 1)` on a `(D, 237)` input (verified by execution in
`tests/test_dt_spectral.py`). In this library's taxonomy that is exactly
the definition of `category="day"` (`dt_day.py`: "reduce `dim=-1` down to
a single value per day, shape `(D, 1)`"), and
`alphamine/core/ops_real_backend.py` derives its `reduces_last_dim` flag
as `category == "day"`, which `core/expression.operand_shape_compatible`
then uses to forbid feeding a collapsed result back into an `in_*`/`day_*`
operator. Tagging them `"in"` would advertise a full-width `(D, M)` output
that they never produce, and the downstream batch engine's 3-D layout
dispatch keys off this tag. So: `"day"`.

## Registration / smoke-test status

All four `energy_ratio_*` ops take REQUIRED non-tensor scalar parameters
(`n`, `m`, `q`) with no defaults, so they do not match any of the four
grammar categories (`unary`/`binary`/`rolling`/`pair_rolling`) that
`ops_real_backend._validate` smoke-tests. They are expected to land in
that module's skip list and stay out of `DT_CATALOG` / the RL search
grammar; they are reached through the general-call path instead. They are
still registered via `@op` so they appear in `operator_dict_DT` with
honest taxonomy metadata.

`fft_with_nan_handling` is a HELPER, not an operator: it returns a
3-tuple, not a tensor. It is intentionally left undecorated so it never
enters `OPERATOR_REGISTRY` / `operator_dict_DT`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.fft

from .common import op

__all__ = [
    "fft_with_nan_handling",
    "energy_ratio_n",
    "energy_ratio_band_n",
    "energy_ratio_topm",
    "energy_ratio_topm_minus_q",
]


# --------------------------------------------------------------------------
# helper (NOT registered: returns a tuple, not a tensor)
# --------------------------------------------------------------------------

def fft_with_nan_handling(
    x: torch.Tensor, pad_to_pow2: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NaN-tolerant power spectrum along the minute axis.

    Returns `(power_spectrum, valid_points, valid_points_f)` where
    `power_spectrum` is `(D, max_len)`, `valid_points` is an int64 `(D, 1)`
    per-row count of non-NaN entries, and `valid_points_f` is its float
    view. Rows that are entirely NaN come back as an all-NaN spectrum row.

    `max_len` is the smallest power of two `>= max(valid_points)` when
    `pad_to_pow2` is True, else `max(valid_points)` itself — a BATCH-WIDE
    quantity, see quirks (1) and (2) in the module docstring.
    """
    device = x.device
    B, L = x.shape

    mask = ~torch.isnan(x)
    x_clean = torch.where(mask, x, torch.zeros_like(x))
    valid_points = mask.sum(dim=1, keepdim=True)  # int64
    valid_points_f = valid_points.float()         # float version for arithmetic

    max_valid = int(valid_points.max().item())
    if max_valid == 0:
        max_len = 1
    else:
        if pad_to_pow2:
            # smallest power of two >= max_valid
            max_len = int(
                2 ** torch.ceil(torch.log2(torch.tensor(max_valid, device=device))).item()
            )
        else:
            max_len = max_valid
    max_len = max(1, max_len)

    # LEGACY QUIRK (2): the `else` branch CROPS to `max_len`, so a batch
    # whose longest row is shorter than L loses columns from every row.
    if L < max_len:
        x_padded = torch.nn.functional.pad(x_clean, (0, max_len - L))
    else:
        x_padded = x_clean[:, :max_len]

    # LEGACY QUIRK (1): this masks by POSITION (`col < valid_points`), not
    # by validity, so interior NaNs stay as zero-holes while an equal
    # number of valid tail samples is discarded.
    col_idx = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
    valid_mask = col_idx < valid_points  # broadcasting -> [B, max_len]

    padded = torch.where(valid_mask, x_padded, torch.zeros_like(x_padded))

    fft_result = torch.fft.fft(padded, dim=1)
    power_spectrum = torch.abs(fft_result) ** 2   # [B, max_len]

    # all-NaN row -> mark the whole row NaN
    power_spectrum = torch.where(
        valid_points_f > 0, power_spectrum, torch.full_like(power_spectrum, float("nan"))
    )

    return power_spectrum, valid_points, valid_points_f


# --------------------------------------------------------------------------
# spectral energy-concentration ratios
# --------------------------------------------------------------------------

@op("dt", "day", arity=1)
def energy_ratio_n(x: torch.Tensor, n: float, pad_to_pow2: bool = True) -> torch.Tensor:
    """Fraction of the spectrum needed to reach `n` of the total energy.

    Sorts each row's power spectrum descending, finds the smallest count
    `k` of bins whose cumulative energy reaches `n * total_energy`, and
    returns `k / F` where `F` is `valid_points` (if `pad_to_pow2`) or the
    FFT length `M` (if not) — see quirk (3). Shape `(D, 1)`.
    """
    power_spectrum, valid_points_int, valid_points = fft_with_nan_handling(x, pad_to_pow2)
    B, M = power_spectrum.shape

    total_energy = power_spectrum.sum(dim=1, keepdim=True)  # [B,1]

    sorted_power, _ = torch.sort(power_spectrum, dim=1, descending=True)
    cum_energy = torch.cumsum(sorted_power, dim=1)  # [B, M]
    threshold = n * total_energy                    # [B, 1]

    # per-row first index where cumulative >= threshold
    ge_mask = cum_energy >= threshold  # [B, M] boolean
    any_ge = ge_mask.any(dim=1)        # [B]

    # argmax gives the first True; rows with no True fall back to M-1
    first_idx = ge_mask.float().argmax(dim=1)  # [B]
    first_idx = first_idx.to(torch.long)

    fallback = torch.full((B,), M - 1, device=x.device, dtype=torch.long)
    idx = torch.where(any_ge, first_idx, fallback)  # [B]

    k = (idx + 1).unsqueeze(1).float()  # [B,1], 1-based count

    # LEGACY QUIRK (3): denominator is a TIME-sample count in one branch
    # and an FFT-BIN count in the other.
    if pad_to_pow2:
        F = valid_points  # float [B,1]
    else:
        F = torch.full_like(valid_points, fill_value=float(M))  # [B,1]

    ratio = k / (F + 1e-12)

    # invalid rows: valid_points == 0 or total_energy == 0 -> NaN
    ratio = torch.where(
        (valid_points > 0) & (total_energy > 0), ratio, torch.full_like(ratio, float("nan"))
    )
    return ratio


@op("dt", "day", arity=1)
def energy_ratio_band_n(
    x: torch.Tensor, n: float, high: bool = True, pad_to_pow2: bool = True
) -> torch.Tensor:
    """`energy_ratio_n` restricted to the high (`high=True`) or low half of
    the spectrum. Shape `(D, 1)`.

    The high band is bins `[N//2:]`, the low band bins `[:(N+1)//2]`.
    Normalization uses `band_valid`, which mixes a bin offset into a
    sample count — see quirk (4).
    """
    power_spectrum, valid_points_int, valid_points = fft_with_nan_handling(x, pad_to_pow2)
    B, N = power_spectrum.shape

    # LEGACY QUIRK (4): `band_valid_int` subtracts/clamps a FREQUENCY-BIN
    # offset against a TIME-SAMPLE count.
    if high:
        band = power_spectrum[:, N // 2:]  # [B, N_high]
        band_len = band.shape[1]
        # band_valid = max(0, valid_points - N//2)
        band_valid_int = torch.clamp(valid_points_int - (N // 2), min=0)
    else:
        band = power_spectrum[:, :(N + 1) // 2]  # [B, N_low]
        band_len = band.shape[1]
        band_valid_int = torch.minimum(
            valid_points_int, torch.tensor((N + 1) // 2, device=x.device)
        )

    band_valid = band_valid_int.float()  # [B,1]

    total_band_energy = band.sum(dim=1, keepdim=True)  # [B,1]
    sorted_band, _ = torch.sort(band, dim=1, descending=True)
    cum_energy = torch.cumsum(sorted_band, dim=1)  # [B, band_len]
    threshold = n * total_band_energy              # [B,1]

    ge_mask = cum_energy >= threshold  # [B, band_len]
    any_ge = ge_mask.any(dim=1)
    first_idx = ge_mask.float().argmax(dim=1).to(torch.long)
    fallback = torch.full((B,), band_len - 1, device=x.device, dtype=torch.long)
    idx = torch.where(any_ge, first_idx, fallback)
    k = (idx + 1).unsqueeze(1).float()  # [B,1]

    ratio = k / (band_valid + 1e-12)

    # invalid row, band_valid == 0, or total_band_energy == 0 -> NaN
    return torch.where(
        (band_valid > 0) & (total_band_energy > 0), ratio, torch.full_like(ratio, float("nan"))
    )


@op("dt", "day", arity=1)
def energy_ratio_topm(x: torch.Tensor, m: int, pad_to_pow2: bool = True) -> torch.Tensor:
    """Share of total spectral energy held by the `m` strongest bins.

    Shape `(D, 1)`. `m` is clamped to the spectrum width.
    """
    power_spectrum, valid_points_int, valid_points = fft_with_nan_handling(x, pad_to_pow2)
    # LEGACY QUIRK (6): all-NaN rows become -inf here and their -inf sum
    # flows into `topm_energy`; the final `torch.where` masks it back to NaN.
    safe_power = torch.where(
        valid_points > 0, power_spectrum, torch.full_like(power_spectrum, -float("inf"))
    )
    m = min(m, power_spectrum.shape[1])
    topm_vals, _ = torch.topk(safe_power, k=m, dim=1)
    topm_energy = topm_vals.sum(dim=1, keepdim=True)
    total_energy = power_spectrum.sum(dim=1, keepdim=True)
    ratio = topm_energy / (total_energy + 1e-12)
    return torch.where(valid_points > 0, ratio, torch.full_like(ratio, float("nan")))


@op("dt", "day", arity=1)
def energy_ratio_topm_minus_q(
    x: torch.Tensor, m: int, q: int, pad_to_pow2: bool = True
) -> torch.Tensor:
    """Share of total spectral energy held by ranks `q..m` of the spectrum
    (the top-`m` bins with the strongest `q` dropped). Shape `(D, 1)`.

    Rows with fewer than `q + 1` valid samples return NaN.
    """
    # LEGACY QUIRK (5): this assert runs BEFORE `m` is clamped to the
    # spectrum width below, so a clamped `m <= q` silently yields an empty
    # slice summing to 0.0 instead of tripping the guard.
    assert m > q, "m must be greater than q"
    power_spectrum, valid_points_int, valid_points = fft_with_nan_handling(x, pad_to_pow2)
    safe_power = torch.where(
        valid_points > 0, power_spectrum, torch.full_like(power_spectrum, -float("inf"))
    )
    m = min(m, power_spectrum.shape[1])
    topm_vals, _ = torch.topk(safe_power, k=m, dim=1)
    remaining = topm_vals[:, q:]  # [B, m-q] (m-q may be 0)
    remaining_energy = remaining.sum(dim=1, keepdim=True)
    total_energy = power_spectrum.sum(dim=1, keepdim=True)
    has_enough = valid_points_int >= (q + 1)
    ratio = remaining_energy / (total_energy + 1e-12)
    return torch.where(
        has_enough & (valid_points > 0), ratio, torch.full_like(ratio, float("nan"))
    )
