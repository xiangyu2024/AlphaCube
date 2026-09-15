"""Tests for `alphacube.models.conformal_risk`:

1. `conditional_value_at_risk_ru` against a hand-computable example and
   against the "average of the worst (1-beta) fraction" equivalent
   characterization (no ties at the VaR threshold).
2. `ConformalCVaRController.step()` over many rounds on a synthetic
   i.i.d.-ish loss sequence, checking `lambda_` stays bounded and the
   realized empirical CVaR converges near `alpha`.
3. An adversarial/regime-switching robustness check: a low-vol regime
   followed by a high-vol/crash regime, verifying `lambda_` (exposure)
   actually drops in the crash regime.
4. `calibrate_from_history` matches manual sequential `.step()` calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacube.models.conformal_risk import (
    ConformalCVaRController,
    conditional_value_at_risk_ru,
)


# ---------------------------------------------------------------------
# conditional_value_at_risk_ru
# ---------------------------------------------------------------------


def test_cvar_ru_hand_computable_example():
    # losses = [0]*8 + [1, 10], beta=0.8. RU objective
    # c + 1/(1-beta)*mean((losses-c)_+) is flat and minimized at value 5.5
    # for any c in [0, 1] (verified analytically: the worst 1-beta=20%
    # tail is exactly {10, 1}, whose average is (10+1)/2 = 5.5, and this
    # matches the RU objective's minimum by the RU<->quantile-tail-average
    # equivalence when the tail fraction lands exactly on a sample boundary).
    losses = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 10], dtype=float)
    beta = 0.8
    result = conditional_value_at_risk_ru(losses, beta)
    assert result == pytest.approx(5.5, abs=1e-6)


def test_cvar_ru_matches_worst_fraction_average_no_ties():
    # losses = 1..10 (no ties), beta=0.7 -> worst 30% = top 3 values
    # {8, 9, 10}, average = 9. No probability mass sits exactly at the
    # quantile boundary in a way that creates ambiguity, so RU CVaR should
    # equal this "average of the worst (1-beta) fraction" characterization.
    losses = np.arange(1, 11, dtype=float)
    beta = 0.7
    result = conditional_value_at_risk_ru(losses, beta)

    k = int(round((1.0 - beta) * len(losses)))
    expected = np.sort(losses)[::-1][:k].mean()
    assert result == pytest.approx(expected, abs=1e-4)
    assert result == pytest.approx(9.0, abs=1e-4)


def test_cvar_ru_constant_losses():
    losses = np.full(10, 3.0)
    assert conditional_value_at_risk_ru(losses, 0.9) == pytest.approx(3.0)


def test_cvar_ru_rejects_invalid_beta():
    losses = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        conditional_value_at_risk_ru(losses, 0.0)
    with pytest.raises(ValueError):
        conditional_value_at_risk_ru(losses, 1.0)


def test_cvar_ru_rejects_empty():
    with pytest.raises(ValueError):
        conditional_value_at_risk_ru(np.array([]), 0.9)


# ---------------------------------------------------------------------
# ConformalCVaRController construction / validation
# ---------------------------------------------------------------------


def test_controller_validates_params():
    with pytest.raises(ValueError):
        ConformalCVaRController(beta=1.5, alpha=0.01, r_min=-1, r_max=1, lambda_min=0, lambda_max=1)
    with pytest.raises(ValueError):
        ConformalCVaRController(beta=0.9, alpha=0.01, r_min=1, r_max=-1, lambda_min=0, lambda_max=1)
    with pytest.raises(ValueError):
        ConformalCVaRController(beta=0.9, alpha=0.01, r_min=-1, r_max=1, lambda_min=1, lambda_max=0)


def test_controller_default_lambda_init_is_midpoint():
    ctrl = ConformalCVaRController(
        beta=0.9, alpha=0.01, r_min=-1.0, r_max=1.0, lambda_min=0.2, lambda_max=0.8
    )
    assert ctrl.lambda_ == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Synthetic loss model shared by the convergence + adversarial tests
# ---------------------------------------------------------------------

R_MIN, R_MAX = -1.0, 1.0
LAMBDA_MIN, LAMBDA_MAX = 0.0, 1.0


def _clipped_lambda(lam: float) -> float:
    return min(max(lam, LAMBDA_MIN), LAMBDA_MAX)


def _realized_loss(lam: float, mu: float, sigma: float, z: float) -> float:
    """R_t = -return_t = -(lambda*mu + lambda*sigma*z), loss-clipped at the
    lambda box edges per the paper's unprojected-controller convention
    (state lambda_t is allowed to drift outside the box; only the loss
    function clips)."""
    lam_eff = _clipped_lambda(lam)
    ret = lam_eff * mu + lam_eff * sigma * z
    loss = -ret
    return float(np.clip(loss, R_MIN, R_MAX))


# ---------------------------------------------------------------------
# ConformalCVaRController.step() convergence test
# ---------------------------------------------------------------------


def test_controller_step_stays_bounded_and_tracks_alpha():
    rng = np.random.default_rng(0)
    alpha = 0.05
    ctrl = ConformalCVaRController(
        beta=0.85,
        alpha=alpha,
        r_min=R_MIN,
        r_max=R_MAX,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX,
        gamma_0=0.05,
    )

    mu, sigma = 0.03, 0.20
    n_rounds = 2000
    realized_losses = []
    lam = ctrl.lambda_
    for _ in range(n_rounds):
        z = rng.standard_normal()
        loss = _realized_loss(lam, mu, sigma, z)
        realized_losses.append(loss)
        lam = ctrl.step(loss)

        # lambda_ must never diverge -- the paper's regret bound requires
        # this to stay within a bounded neighborhood of the box, not go to
        # +-infinity, even though it's allowed to drift outside the box.
        assert -10.0 <= lam <= 10.0

    realized_losses = np.array(realized_losses)
    empirical_cvar = conditional_value_at_risk_ru(realized_losses, ctrl.beta)

    # Core guarantee, checked empirically: realized CVaR should end up near
    # the target alpha (within a generous but real tolerance -- this is a
    # T^(-1/2) rate bound, not exact convergence).
    assert empirical_cvar == pytest.approx(alpha, abs=0.05)

    # current_cvar_estimate() (running mean of ell_t_RU) should also be in
    # a sane range, in the same ballpark as alpha.
    assert abs(ctrl.current_cvar_estimate() - alpha) < 0.2


def test_controller_history_tracks_rounds():
    ctrl = ConformalCVaRController(
        beta=0.9, alpha=0.02, r_min=R_MIN, r_max=R_MAX, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX
    )
    rng = np.random.default_rng(1)
    n_rounds = 25
    for _ in range(n_rounds):
        ctrl.step(float(rng.uniform(R_MIN, R_MAX)))

    assert ctrl.t == n_rounds
    assert len(ctrl.history) == n_rounds
    df = ctrl.history_df()
    assert len(df) == n_rounds
    for col in ("t", "c_t", "lambda_t", "realized_loss", "ell_t_RU"):
        assert col in df.columns


# ---------------------------------------------------------------------
# Adversarial / regime-switching robustness test
# ---------------------------------------------------------------------


def test_controller_adapts_down_in_crash_regime():
    """The paper's actual selling point: the controller should reduce
    exposure (lambda) when a sudden high-volatility/crash regime hits,
    without any i.i.d. assumption baked into the update rule itself."""
    rng = np.random.default_rng(42)
    alpha = 0.03
    ctrl = ConformalCVaRController(
        beta=0.85,
        alpha=alpha,
        r_min=R_MIN,
        r_max=R_MAX,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX,
        gamma_0=0.05,
    )

    n_calm = 1000
    n_crash = 1000
    mu_calm, sigma_calm = 0.05, 0.08
    mu_crash, sigma_crash = -0.10, 0.60  # sudden regime switch: negative drift, high vol

    calm_lambdas = []
    crash_lambdas = []
    lam = ctrl.lambda_

    for _ in range(n_calm):
        z = rng.standard_normal()
        loss = _realized_loss(lam, mu_calm, sigma_calm, z)
        lam = ctrl.step(loss)
        calm_lambdas.append(lam)

    for _ in range(n_crash):
        z = rng.standard_normal()
        loss = _realized_loss(lam, mu_crash, sigma_crash, z)
        lam = ctrl.step(loss)
        crash_lambdas.append(lam)

    mean_calm = float(np.mean(calm_lambdas[-200:]))  # tail of calm regime, post warm-up
    mean_crash = float(np.mean(crash_lambdas[-200:]))  # tail of crash regime, post adaptation

    # The exposure controller must adapt DOWN once the crash regime hits.
    assert mean_crash < mean_calm, (
        f"expected lower mean lambda in crash regime, got calm={mean_calm:.4f} "
        f"crash={mean_crash:.4f}"
    )
    # A real, not marginal, effect.
    assert mean_calm - mean_crash > 0.1


# ---------------------------------------------------------------------
# calibrate_from_history
# ---------------------------------------------------------------------


def test_calibrate_from_history_matches_manual_steps():
    rng = np.random.default_rng(7)
    realized_losses = list(rng.uniform(R_MIN, R_MAX, size=50))

    ctrl_batch = ConformalCVaRController(
        beta=0.9, alpha=0.02, r_min=R_MIN, r_max=R_MAX, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX
    )
    final_lambda = ctrl_batch.calibrate_from_history(realized_losses)

    ctrl_manual = ConformalCVaRController(
        beta=0.9, alpha=0.02, r_min=R_MIN, r_max=R_MAX, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX
    )
    for loss in realized_losses:
        ctrl_manual.step(loss)

    assert final_lambda == pytest.approx(ctrl_manual.lambda_)
    assert ctrl_batch.lambda_ == pytest.approx(ctrl_manual.lambda_)
    assert ctrl_batch.c == pytest.approx(ctrl_manual.c)
    assert ctrl_batch.t == ctrl_manual.t == 50


def test_calibrate_from_history_rejects_mismatched_lambdas_length():
    ctrl = ConformalCVaRController(
        beta=0.9, alpha=0.02, r_min=R_MIN, r_max=R_MAX, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX
    )
    with pytest.raises(ValueError):
        ctrl.calibrate_from_history([0.1, 0.2, 0.3], lambdas=[0.5, 0.5])
