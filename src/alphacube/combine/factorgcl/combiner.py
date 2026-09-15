"""FactorGCLCombiner: a downstream nonlinear factor-combination / return-
prediction model -- NOT an expression miner. Trains `FactorGCLModel` jointly
on MSE(forward returns) + a temporal residual contrastive auxiliary loss
(`contrastive.py`) on top of already-mined factor panels (e.g. a
`mining.pool` pool's `z_columns()`).

Interface is deliberately parallel to (but separate from) the pools in
`mining/pool.py`: `fit(panels, target, epochs)` / `predict(panels) ->
np.ndarray`, since this produces a learned nonlinear combination rather than
linear pool weights. Already-computed `(n_days, n_instruments)` panels are
the only input -- the previous project's version could also take a pool and
a data object and evaluate the member expressions itself; the cube engine
now does that upstream.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .contrastive import temporal_residual_infonce
from .hypergraph import build_prior_incidence
from .model import FactorGCLModel


class FactorGCLCombiner:
    def __init__(
        self,
        embed_dim: int = 16,
        n_hidden_factors: int = 8,
        lr: float = 1e-3,
        contrastive_coef: float = 0.1,
        contrastive_gap: int = 5,
        max_days_per_epoch: int = 40,
        seed: int = 0,
    ) -> None:
        self.embed_dim = embed_dim
        self.n_hidden_factors = n_hidden_factors
        self.lr = lr
        self.contrastive_coef = contrastive_coef
        self.contrastive_gap = contrastive_gap
        self.max_days_per_epoch = max_days_per_epoch

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.model: Optional[FactorGCLModel] = None
        self.loss_history: List[float] = []
        self._n_factors_effective = 1
        self._fit_shape: Optional[Tuple[int, int]] = None

    @staticmethod
    def _prior_from_panels(
        panels: Sequence[np.ndarray], shape: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """`build_prior_incidence(panels)`, or -- with no panels at all -- a
        single all-zero column of the given `(n_days, n_instruments)` shape,
        since the model always needs at least one input feature (the
        previous project did the same for an empty pool)."""
        if len(panels) == 0:
            if shape is None:
                raise ValueError("no panels given and no (n_days, n_instruments) shape to fall back on")
            return np.zeros(tuple(shape) + (1,), dtype=np.float64)
        return build_prior_incidence(panels)

    def fit(
        self,
        panels: Sequence[np.ndarray],
        target: np.ndarray,
        epochs: int = 5,
        *,
        train_days: Optional[slice] = None,
    ) -> List[float]:
        """Train on already-computed factor panels against `target`.

        `panels` is a sequence of `(n_days, n_instruments)` factor panels --
        raw or already z-scored, since `build_prior_incidence` z-scores each
        one per day and that is idempotent; `target` is the label panel of
        the same shape, e.g. `DailyData.forward_returns(horizon)`. No panels
        at all is allowed: a single all-zero input column stands in, so the
        model degenerates to its hidden-factor and residual stages.

        `train_days` restricts which days the optimizer ever sees. Without it
        `fit` trains on the whole panel, so any later evaluation is in-sample
        and means nothing -- `combine.post_combine` always passes the training
        slice.
        """
        target = np.asarray(target, dtype=np.float64)
        if target.ndim != 2:
            raise ValueError(f"target must be (n_days, n_instruments), got shape {target.shape}")
        prior = self._prior_from_panels(panels, target.shape)
        if prior.shape[:2] != target.shape:
            raise ValueError(f"panels have shape {prior.shape[:2]}, target has shape {target.shape}")
        self._fit_shape = (int(target.shape[0]), int(target.shape[1]))
        self._n_factors_effective = prior.shape[-1]

        self.model = FactorGCLModel(
            n_prior_factors=self._n_factors_effective,
            embed_dim=self.embed_dim,
            n_hidden_factors=self.n_hidden_factors,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        n_days = prior.shape[0]

        finite_rows = np.all(np.isfinite(prior), axis=-1) & np.isfinite(target)  # (T, N)
        gap = self.contrastive_gap
        lo, hi = 0, n_days
        if train_days is not None:
            lo, hi, _ = train_days.indices(n_days)
        valid_days = [
            t
            for t in range(lo, max(lo, min(hi, n_days) - gap))
            if finite_rows[t].sum() >= 2 and finite_rows[t + gap].sum() >= 2
        ]

        self.loss_history = []
        if not valid_days:
            return self.loss_history

        prior_t = torch.tensor(np.nan_to_num(prior, nan=0.0), dtype=torch.float32)
        target_t = torch.tensor(np.nan_to_num(target, nan=0.0), dtype=torch.float32)
        finite_t = torch.tensor(finite_rows)

        for _ in range(epochs):
            n_sample = min(self.max_days_per_epoch, len(valid_days))
            days = self._rng.choice(valid_days, size=n_sample, replace=False)
            epoch_losses = []

            for t in days:
                t = int(t)
                t2 = t + gap
                pred1, resid1, _ = self.model(prior_t[t])
                _, resid2, _ = self.model(prior_t[t2])

                mask1 = finite_t[t]
                mse = F.mse_loss(pred1[mask1], target_t[t][mask1])

                common_mask = finite_t[t] & finite_t[t2]
                if int(common_mask.sum()) >= 2:
                    contrastive = temporal_residual_infonce(resid1[common_mask], resid2[common_mask])
                else:
                    contrastive = torch.zeros(())

                loss = mse + self.contrastive_coef * contrastive
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            self.loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

        return self.loss_history

    def predict(self, panels: Sequence[np.ndarray]) -> np.ndarray:
        """Predict over the whole panel. `panels` mirrors `fit`'s (the same
        factors in the same order, over any number of days); with no panels
        the `(n_days, n_instruments)` seen at fit time is reused."""
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        prior = self._prior_from_panels(panels, self._fit_shape)
        if prior.shape[-1] != self._n_factors_effective:
            raise ValueError(
                f"predict got {prior.shape[-1]} panel(s) but fit saw {self._n_factors_effective}"
            )
        n_days, n_instruments = prior.shape[0], prior.shape[1]
        prior_t = torch.tensor(np.nan_to_num(prior, nan=0.0), dtype=torch.float32)

        self.model.eval()
        preds = np.zeros((n_days, n_instruments), dtype=np.float64)
        with torch.no_grad():
            for t in range(n_days):
                pred, _, _ = self.model(prior_t[t])
                preds[t] = pred.numpy()
        self.model.train()
        return preds
