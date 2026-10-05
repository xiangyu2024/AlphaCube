"""股票池掩码:(D, S) bool,和行情面板(tradable_backtest prepare 的 npz)同一个日期、股票顺序。

现在只有一种:沪深300 近似。立方体里没有指数成分股,用 daily_capital 的市值搭(规则同
tearsheet.index_members):第一天和每年 1 月、7 月第一个交易日调样,上市满 90 天、过去 250 天日均成交额
排前 50% 的票里,取过去 250 天日均总市值最大的 300 只。真实沪深300 在 6 月、12 月调样,用自由流通市值,
这里是近似。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/universe.py \\
      --out _bench/universe/hs300_proxy.npy
  # 反过来:每天不在某个名单里的票("专门不买沪深300")
  python scripts/universe.py --complement-of _bench/universe/hs300_proxy_v1.npy --out _bench/universe/non_hs300_v1.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_universe(path: Optional[str], shape) -> Optional[np.ndarray]:
    """读掩码并核对形状;path 为空返回 None(= 全市场)。"""
    if not path:
        return None
    u = np.load(path)
    if u.shape != tuple(shape) or u.dtype != bool:
        raise SystemExit(f"股票池 {path} 是 {u.shape} {u.dtype},要的是 {tuple(shape)} bool")
    return u


def complement(src: str, out: str) -> int:
    """名单的补集:每天不在 src 里的票都算进来(还没上市、停牌的格子也在里面 —— 回测自己会跳过买不了的)。"""
    import json

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    u = np.load(src)
    if u.dtype != bool:
        raise SystemExit(f"{src} 不是 bool 掩码")
    np.save(out, ~u)
    meta_src = Path(src).with_suffix(".json")
    meta = json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.exists() else {}
    meta["rule"] = f"{src} 的补集(每天不在名单里的票)"
    Path(out).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"{out}:{src} 的补集,每天 {(~u).sum(1).min()}~{(~u).sum(1).max()} 只(含还没上市的格子)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="_bench/tradable/market_20180102_20251231.npz")
    ap.add_argument("--capital-dir", default="/root/autodl-tmp/qmt_daily_data/daily_capital")
    ap.add_argument("--amount", default="_bench/tradable/amount_day.npy")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--complement-of", default="", help="不重新搭名单,只写这个名单的补集")
    a = ap.parse_args(argv)
    if a.complement_of:
        return complement(a.complement_of, a.out)

    import tradable_backtest as tb
    import tearsheet as ts

    m = tb.load_market(a.inputs, fill=False)
    cmv, tmv = ts.load_caps(a.capital_dir, m.dates, m.symbols)
    member = ts.index_members(tmv, np.load(a.amount), m.list_day, m.dates, n=a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, member)
    # 日期和股票顺序另存一份:挖掘(alphacube mine --universe)按名字对齐,不靠位置
    import json
    Path(a.out).with_suffix(".json").write_text(json.dumps({"dates": list(m.dates), "symbols": list(m.symbols),
                                                            "rule": f"沪深300 近似,前 {a.n} 只"}, ensure_ascii=False),
                                                encoding="utf-8")
    counts = member.sum(1)
    changes = [(m.dates[t], int((member[t] & ~member[t - 1]).sum())) for t in range(1, len(m.dates))
               if (member[t] != member[t - 1]).any()]
    print(f"{a.out}:每天 {counts.min()}~{counts.max()} 只,{len(m.dates)} 天里进过池子的共 {int(member.any(0).sum())} 只;"
          f"调样 {len(changes)} 次,每次换入 {min(c for _, c in changes)}~{max(c for _, c in changes)} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
