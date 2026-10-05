"""build_champion 选出的族 -> RL 池子种子(``--seed-library`` 读的 yaml),每族一个代表,按族的训练段适应度排。

build_champion 的 ``<out>.families.json`` 里族已经按"族内最好的 |训练段 rank IC| - 0.14 x 换手"排好,
每族成员也按这个排,所以每族第一个就是代表。种子按文件顺序逐个尝试进池子(去重、容量照常),排前面的先占位。

  PYTHONPATH=src python scripts/seed_yaml.py --families _bench/legacy_signal_hs300.families.json \\
      --manifest _bench/legacy_hs300/manifest.csv --top 50 --out _bench/universe/legacy_seeds_hs300.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence


def representatives(families: List[dict], manifest: List[dict], top: int) -> List[dict]:
    by_i = {int(r["i"]): r for r in manifest}
    out = []
    for fam in families[: top or None]:
        r = by_i[int(fam["members"][0])]
        out.append({"name": r.get("name") or f"f{r['i']}", "expression": r["expression"]})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fams = json.loads(Path(a.families).read_text(encoding="utf-8"))
    man = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    seeds = representatives(fams, man, a.top)
    Path(a.out).write_text(yaml.safe_dump({"seed_factors": seeds}, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    print(f"{len(seeds)} 个族代表写到 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
