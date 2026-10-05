"""几个合成信号逐日转成秩 z 分数后等权相加(不拟合权重,测试段无从挑选),存成新信号给 tradable_backtest。

  PYTHONPATH=src python scripts/blend_signals.py --signal _bench/legacy_signal.npy \\
      --signal _bench/champion_signal_buyable.npy --out _bench/blend_legacy_mined.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def blend(signals: List[np.ndarray]) -> np.ndarray:
    """逐日秩 z 分数的等权平均;某只票某天只要有一个信号有值就给分(缺的那个不参与平均)。"""
    from alphacube.metrics import rank_rows, zscore_rows

    zs = np.stack([zscore_rows(rank_rows(np.asarray(s, dtype=np.float64))) for s in signals])
    n = np.isfinite(zs).sum(0)
    out = np.where(n > 0, np.nansum(zs, 0) / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sigs = [np.load(p) for p in a.signal]
    if len({s.shape for s in sigs}) != 1:
        raise SystemExit(f"信号形状不一致:{[s.shape for s in sigs]}")
    np.save(a.out, blend(sigs))
    print(f"{len(sigs)} 个信号等权合成 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
