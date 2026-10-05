"""in_mask_nearby_ones 向量化前后逐位一致(旧实现按元素循环,全市场上要跑好几天)。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from alphacube.operators.dt_in import in_mask_nearby_ones


def _reference(x: torch.Tensor, d: int) -> torch.Tensor:
    result = x.clone()
    ones = torch.nonzero(~torch.isnan(x))
    if ones.numel() == 0:
        return result
    _, idx = torch.sort(ones[:, 0] * x.shape[1] + ones[:, 1])
    sorted_ones = ones[idx]
    last_anchor = torch.full((x.shape[0],), -(10 ** 9), dtype=torch.long)
    for i in range(sorted_ones.size(0)):
        b, pos = int(sorted_ones[i, 0]), int(sorted_ones[i, 1])
        if pos > last_anchor[b] + d:
            last_anchor[b] = pos
        else:
            result[b, pos] = float("nan")
    return result


@pytest.mark.parametrize("d", [0, 1, 3, 10])
@pytest.mark.parametrize("density", [0.0, 0.1, 0.5, 1.0])
def test_matches_the_elementwise_reference(d, density):
    rng = np.random.default_rng(d * 7 + int(density * 10))
    x = np.where(rng.random((40, 37)) < density, 1.0, np.nan).astype(np.float32)
    t = torch.from_numpy(x)
    got, want = in_mask_nearby_ones(t, d), _reference(t, d)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    assert torch.equal(torch.nan_to_num(got, nan=-7.0), torch.nan_to_num(want, nan=-7.0))


def test_keeps_values_and_rejects_negative_window():
    t = torch.tensor([[1.0, 1.0, float("nan"), 1.0, 1.0, 1.0]])
    assert torch.equal(torch.isnan(in_mask_nearby_ones(t, 1)), torch.tensor([[False, True, True, False, True, False]]))
    with pytest.raises(ValueError):
        in_mask_nearby_ones(t, -1)
