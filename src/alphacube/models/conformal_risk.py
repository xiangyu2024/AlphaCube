"""Online, distribution-free CVaR control via Rockafellar-Uryasev conformal
inference (Chen, Shen, Deng, Lei, "Adversarially Robust Control of
Conditional Value-at-Risk via Rockafellar-Uryasev Conformal Inference",
arXiv:2606.00320).

Adjacent to `models/evaluation.py` (which scores *fixed* prediction panels
after the fact) and `metrics.py` (which scores mined expression panels
against a forward-return target during mining): this module is
neither. It's a *streaming controller* — given a sequence of realized
losses that depend on a decision you make each round, it tells you how to
adjust that decision, round by round, so the empirical CVaR of the realized
losses tracks a target level. There's no train/test split, no batch
re-fitting: `.step()` is called once per round as new data arrives.

**Why this is not the usual (split-)conformal recipe.** Standard conformal
prediction calibrates a threshold once on a held-out i.i.d./exchangeable
calibration set and then applies it unchanged. That breaks for financial
return sequences, which are neither i.i.d. nor exchangeable (volatility
clusters, regimes shift, crises are exactly the moments a static threshold
is wrong). This module instead runs two nested *online* (regret-minimizing)
loops that keep re-calibrating every round, and the resulting guarantee
holds for ANY sequence of losses, including adversarially chosen ones — see
`ConformalCVaRController`'s docstring for the exact statement.

**The Rockafellar-Uryasev (RU) representation of CVaR.** For a loss sample
`R_1, ..., R_T` and confidence level `beta in (0,1)`:

    CVaR_beta(R_1:T) = min_c { c + 1/(1-beta) * (1/T) * sum_t (R_t - c)_+ }

where `(x)_+ = max(x, 0)`. The minimizing `c` is (a) VaR_beta, the
`beta`-quantile of the losses; the minimum value itself is CVaR_beta, the
mean loss in the worst `(1-beta)` tail. This one identity is what makes
CVaR estimation an ordinary (if non-smooth) 1-D convex minimization instead
of a quantile-then-tail-average two-step procedure, and it's what both the
batch estimator (`conditional_value_at_risk_ru`) and the online controller
(`ConformalCVaRController`) are built on.

**Suggested project integration** (not implemented here, since this module
must not import mining/eval-specific code): a natural loss series for
`ConformalCVaRController` is `R_t = -portfolio_return_t`, so that `alpha`
becomes an explicit target downside-risk budget (e.g. "hold realized CVaR
of daily loss at or below 1%") and `lambda_t` plays the role of a dynamic
gross/long exposure level, directly analogous to the previous project's
ad hoc breadth-based dynamic-exposure schedule — except
here the exposure adjustment comes with a formal worst-case CVaR bound
instead of a hand-tuned threshold. `R_t(lambda)` itself would typically be
built from `models/targets.py`'s forward-return/event labels (e.g. scaling
a `forward_event_label(..., kind="crash")`-flagged tail by `lambda`), or
from a crash-probability signal turned into a realized P&L series — this
module
intentionally treats `R_t(.)` as an opaque black box (per the paper's own
treatment) and never imports either of those modules itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import threadpoolctl
from scipy.optimize import minimize_scalar

# Every `minimize_scalar` call below is wrapped in
# `threadpoolctl.threadpool_limits(1)`. This is a defensive workaround for
# an environment-specific issue on this machine: when scipy's bundled
# BLAS backend spins up its own multi-threaded pool in the same process
# that also uses the cube engine's torch-based kernels (`engine/`), the two
# thread pools can collide and segfault the interpreter -- reproducible in
# the previous project simply by running this module's tests alongside its
# torch-operator tests in the same pytest session without this guard.
# Limiting BLAS to a single
# thread only for the (cheap, 1-D, already-fast) duration of each convex
# solve avoids the collision without changing anything outside this file.


def conditional_value_at_risk_ru(losses: np.ndarray, beta: float) -> float:
    """Plain (batch, non-online) Rockafellar-Uryasev CVaR estimator.

    Solves ``min_c { c + 1/(1-beta) * mean((losses - c)_+) }`` over a single
    finite sample via bounded 1-D convex minimization (`c` restricted to
    `[losses.min(), losses.max()]`, which always contains a minimizer since
    the RU objective's slope is `1 - 1/(1-beta) <= 0` below any loss value
    still in the tail and `>= 0`... in general >0 above `losses.max()`, so
    the minimum cannot lie strictly outside that range).

    This is a plain evaluation METRIC, not part of the online controller
    below — e.g. "CVaR_0.95 of a strategy's daily losses" as a magnitude
    tail-risk statistic, independent of any exposure-control loop. It is a
    natural candidate addition to `models/evaluation.py`'s `evaluate_target`
    magnitude-metrics group (#5, currently bias/RMSE/calibration slope) for
    a future engineer -- e.g. `CVaR_beta(-realized_returns)` alongside
    Sharpe as a tail-risk complement to a purely second-moment-based risk
    view -- but per this module's own scope, `evaluation.py` is not modified
    here.

    Also equals, when there is no probability mass exactly at the
    `beta`-quantile (no ties at the VaR threshold), the plain "average of
    the worst `(1-beta)` fraction of losses" -- verified as a cross-check in
    `tests/test_conformal_risk.py`, not relied on internally (the RU
    formulation is used directly since it, unlike the quantile-then-average
    recipe, is what generalizes to the online setting below).

    Parameters
    ----------
    losses : array of realized losses (higher = worse), not returns.
    beta : CVaR confidence level in (0, 1), e.g. 0.95 for the worst 5% tail.

    Returns
    -------
    The scalar CVaR_beta estimate.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        raise ValueError("losses must be non-empty")
    if not np.all(np.isfinite(losses)):
        raise ValueError("losses must be finite")

    lo, hi = float(losses.min()), float(losses.max())
    if lo == hi:
        # Degenerate (constant) sample: c + 1/(1-beta)*0 is minimized at
        # c = lo = hi, which is also the answer any solver would find, but
        # minimize_scalar needs bounds with lo < hi so short-circuit here.
        return lo

    def objective(c: float) -> float:
        return c + (1.0 / (1.0 - beta)) * float(np.mean(np.maximum(losses - c, 0.0)))

    with threadpoolctl.threadpool_limits(1):
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    return float(result.fun)


@dataclass
class ConformalCVaRController:
    """Online, distribution-free CVaR controller (Rockafellar-Uryasev
    conformal inference, arXiv:2606.00320).

    Runs two nested online loops, updated once per round via `.step()`:

    **Inner loop (AdaGrad-FTRL on the RU threshold `c_t`).** Tracks the RU
    threshold that makes `c + 1/(1-beta) * mean((R - c)_+)` small, using the
    per-round surrogate `ell_t_RU(c) = c + 1/(1-beta) * (R_t - c)_+` with
    subgradient `g_t = 1 - 1/(1-beta) * 1[R_t > c_t]`. `q_t` is the running
    AdaGrad sum-of-squared-subgradients accumulator (`q_0 = max(1,
    beta/(1-beta))^2`, matching the paper's initialization so the first
    step size is never larger than the subgradient's own worst-case scale),
    `eta_t = 1/(2*sqrt(q_t))` the resulting adaptive step size, and
    `c_{t+1}` is the exact FTRL (follow-the-regularized-leader) minimizer of
    the accumulated surrogate losses plus an `eta_t`-scaled quadratic
    anchored at the interval midpoint, solved by 1-D bounded convex
    minimization each round (cheap: `scipy.optimize.minimize_scalar` on a
    strictly monotone-slope objective).

    **Outer loop (unprojected online gradient controller on `lambda_t`).**
    `lambda_{t+1} = lambda_t - gamma * (ell_t_RU(c_t) - alpha)`, `gamma =
    gamma_0 * (lambda_max - lambda_min)`. This nudges the decision/exposure
    parameter down when the current round's RU surrogate loss exceeds the
    target `alpha`, up when it's under. Critically, `lambda_t` itself is
    **never clipped/projected back into `[lambda_min, lambda_max]`** -- it
    is allowed to drift outside the box between rounds. Only the *loss
    function*, when evaluated, is clipped: conceptually
    `R_t(lambda) = R_min` for `lambda < lambda_min`, `R_max` for `lambda >
    lambda_max`, and the true model in between. This "unprojected
    controller" design is not an oversight -- keeping the controller state
    unprojected is exactly what makes the paper's regret bound go through
    for adversarial (non-i.i.d., non-exchangeable) sequences; projecting
    `lambda_t` back onto the box every round breaks the telescoping-sum
    argument the regret bound relies on. (This class doesn't clip anything
    itself: `R_t(lambda_t)` is computed by the CALLER, who is responsible
    for applying that clipped-loss convention if their loss function needs
    `lambda` to make sense outside `[lambda_min, lambda_max]`; this class
    only ever sees the resulting scalar `realized_loss_at_lambda`.)

    **Guarantee.** For ANY sequence of realized losses `R_1, ..., R_T`
    (adaptively/adversarially chosen -- no i.i.d. or exchangeability
    assumption needed, which is what makes this suitable for non-i.i.d.
    financial return sequences), the empirical CVaR of the realized
    sequence satisfies

        CVaR_beta(R_1:T) <= alpha + C * T^(-1/2)

    for a constant `C` depending only on `(beta, alpha, gamma, lambda_min,
    lambda_max, R_min, R_max)` -- NOT on the sequence itself. This is a
    deterministic worst-case bound (an online-regret argument), not a
    probabilistic "with probability 1-delta" statement: it holds for every
    realization, including the worst one an adversary could construct.

    Parameters
    ----------
    beta : CVaR confidence level in (0, 1) for the RU threshold (e.g. 0.85).
    alpha : target CVaR level the controller tries to hold the realized
        empirical CVaR at or below (e.g. 0.01).
    r_min, r_max : bounds on the realized-loss range `R_t(.)` can take
        (after loss-clipping at the lambda box edges) -- the closed
        interval the inner loop's `c` is optimized over.
    lambda_min, lambda_max : the decision/exposure box used only to (a)
        compute `gamma` and (b) as the caller's convention for clipping the
        loss function outside the box; `lambda_` itself is not clipped to
        this box (see class docstring).
    gamma_0 : outer-loop base step-size multiplier; `gamma = gamma_0 *
        (lambda_max - lambda_min)`.
    lambda_init : initial decision value; defaults to the box midpoint.

    Attributes
    ----------
    c : current RU threshold estimate (VaR_beta-like quantity).
    q : AdaGrad accumulator for the inner loop.
    lambda_ : current decision/exposure value (unprojected).
    t : number of completed rounds.
    history : list of per-round diagnostic dicts; see `.history_df()`.
    """

    beta: float
    alpha: float
    r_min: float
    r_max: float
    lambda_min: float
    lambda_max: float
    gamma_0: float = 0.05
    lambda_init: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if not self.r_min < self.r_max:
            raise ValueError(f"r_min must be < r_max, got r_min={self.r_min}, r_max={self.r_max}")
        if not self.lambda_min < self.lambda_max:
            raise ValueError(
                f"lambda_min must be < lambda_max, got lambda_min={self.lambda_min}, "
                f"lambda_max={self.lambda_max}"
            )

        if self.lambda_init is None:
            self.lambda_init = (self.lambda_min + self.lambda_max) / 2.0
        self.lambda_ = float(self.lambda_init)

        # RU threshold c starts at r_min rather than the interval midpoint:
        # c is a VaR-like quantile threshold, and r_min is the natural
        # "no losses have exceeded this yet" starting point -- it makes the
        # very first round's subgradient g_1 = 1 - 1/(1-beta) (since any
        # R_1 >= r_min triggers the indicator whenever R_1 > r_min, and
        # equals r_min only in the boundary case), which biases c upward
        # immediately if the first observed loss is high, rather than
        # requiring several rounds to climb up from a possibly-too-low
        # midpoint guess. Either r_min or the midpoint is a defensible
        # choice (both are within the feasible box, so convergence isn't
        # order-sensitive); r_min was picked here for that "start
        # optimistic about tail risk, correct upward on evidence" reading.
        self.c: float = float(self.r_min)

        # AdaGrad accumulator initialization per the paper's spec:
        # q_0 = max(1, beta/(1-beta))^2, guaranteeing the first step size
        # eta_1 = 1/(2*sqrt(q_0)) is never larger than what the
        # subgradient's own worst-case magnitude, max(1, beta/(1-beta)),
        # would warrant.
        self.q: float = max(1.0, self.beta / (1.0 - self.beta)) ** 2

        self.t: int = 0
        self._cvar_running_mean: float = 0.0
        self.history: List[dict] = []

        self._gamma = self.gamma_0 * (self.lambda_max - self.lambda_min)
        self._midpoint = (self.r_min + self.r_max) / 2.0

    # -- internals ---------------------------------------------------

    def _ell_ru(self, c: float, realized_loss: float) -> float:
        """RU surrogate loss `ell_RU(c) = c + 1/(1-beta) * (R - c)_+`."""
        return c + (1.0 / (1.0 - self.beta)) * max(realized_loss - c, 0.0)

    def _solve_inner_ftrl(self, eta_t: float, past_losses: Sequence[float]) -> float:
        """`argmin_{c in [r_min, r_max]} (1/(2*eta_t))*(c - midpoint)^2 +
        sum_{s<=t} ell_s_RU(c)`, solved numerically (convex, 1-D)."""
        losses_arr = np.asarray(past_losses, dtype=np.float64)

        def objective(c: float) -> float:
            quad = (1.0 / (2.0 * eta_t)) * (c - self._midpoint) ** 2
            surrogate_sum = np.sum(c + (1.0 / (1.0 - self.beta)) * np.maximum(losses_arr - c, 0.0))
            return quad + float(surrogate_sum)

        with threadpoolctl.threadpool_limits(1):
            result = minimize_scalar(objective, bounds=(self.r_min, self.r_max), method="bounded")
        return float(result.x)

    # -- public API ----------------------------------------------------

    def step(self, realized_loss_at_lambda: float) -> float:
        """Perform one round of both inner (RU threshold) and outer
        (decision/exposure) updates given the ACTUAL realized loss
        `R_t(lambda_t)` observed this round (for whatever `lambda_t` -- i.e.
        `self.lambda_` before this call -- was actually in effect).

        This class is loss-function-agnostic (per the paper's black-box
        treatment of `R_t(.)`): it doesn't know or need to know how
        `realized_loss_at_lambda` was computed, only its value.

        Returns
        -------
        The new `lambda_{t+1}` to use as the decision/exposure value for
        the next round.
        """
        realized_loss_at_lambda = float(realized_loss_at_lambda)
        c_t = self.c
        lambda_t = self.lambda_

        # RU surrogate loss and subgradient at the CURRENT threshold c_t,
        # evaluated on this round's realized loss.
        ell_t = self._ell_ru(c_t, realized_loss_at_lambda)
        indicator = 1.0 if realized_loss_at_lambda > c_t else 0.0
        g_t = 1.0 - (1.0 / (1.0 - self.beta)) * indicator

        # Inner loop: AdaGrad accumulator + step size, then FTRL solve for
        # c_{t+1} over all surrogate losses seen so far (including this
        # round's).
        self.q = self.q + g_t ** 2
        eta_t = 1.0 / (2.0 * np.sqrt(self.q))

        self.t += 1
        self.history.append(
            {
                "t": self.t,
                "c_t": c_t,
                "lambda_t": lambda_t,
                "realized_loss": realized_loss_at_lambda,
                "ell_t_RU": ell_t,
                "g_t": g_t,
                "eta_t": eta_t,
            }
        )
        past_losses = [h["realized_loss"] for h in self.history]
        c_next = self._solve_inner_ftrl(eta_t, past_losses)

        # Running mean of ell_t_RU -- the batch-averaged RU objective this
        # streaming procedure is implicitly minimizing; see
        # `.current_cvar_estimate()`.
        self._cvar_running_mean += (ell_t - self._cvar_running_mean) / self.t

        # Outer loop: unprojected controller update. lambda_t itself is
        # NOT clipped to [lambda_min, lambda_max] -- see class docstring.
        lambda_next = lambda_t - self._gamma * (ell_t - self.alpha)

        self.c = c_next
        self.lambda_ = lambda_next
        return lambda_next

    def calibrate_from_history(
        self,
        realized_losses: Sequence[float],
        lambdas: Optional[Sequence[float]] = None,
    ) -> float:
        """Batch-replay a historical sequence of realized losses through
        `.step()`, sequentially, as the practical entry point for "I have a
        calibration/backtest history, give me a starting `(c, lambda)`
        state before going live" -- the paper's own suggested use of an
        initial history to warm-start the online procedure before it runs
        on live/streaming data.

        `.step()` itself doesn't consume `lambda`, only the realized loss
        that resulted from whatever `lambda` was in effect -- so `lambdas`
        here is purely a documentation/bookkeeping convenience (recorded
        nowhere but implicitly consistent with each `realized_losses[i]`);
        it is accepted (and length-checked against `realized_losses` if
        given) to make the "the historical lambda varied" case explicit at
        the call site, but if omitted, a fixed historical lambda is assumed
        and simply isn't threaded through the replay.

        Returns the final calibrated `lambda_` after replaying the full
        history.
        """
        realized_losses = list(realized_losses)
        if lambdas is not None and len(lambdas) != len(realized_losses):
            raise ValueError(
                f"lambdas (len {len(lambdas)}) must match realized_losses (len {len(realized_losses)})"
            )
        for loss in realized_losses:
            self.step(loss)
        return self.lambda_

    def current_cvar_estimate(self) -> float:
        """Current running Rockafellar-Uryasev CVaR estimate.

        `self.c` alone is the RU-optimal VaR-like threshold implied by the
        rounds seen so far (at convergence, the minimizer of the RU
        objective); the CVaR value itself is the RU objective's MINIMUM
        VALUE, i.e. the running average of `ell_t_RU(c_t)` across rounds
        (`self._cvar_running_mean`, updated incrementally in `.step()`).
        This method returns that running mean, which is the quantity
        directly comparable to `alpha` and to `conditional_value_at_risk_ru`
        applied post-hoc to the realized-loss history.
        """
        return self._cvar_running_mean

    def history_df(self) -> pd.DataFrame:
        """Per-round diagnostic history (`c_t`, `lambda_t`, `realized_loss`,
        `ell_t_RU`, plus the subgradient/step-size internals) as a
        DataFrame, for plotting/debugging."""
        return pd.DataFrame(self.history)
