"""只读体检:立方体 store 能不能读、读到的东西健不健康。

    PYTHONPATH=src python scripts/check_store.py --store <计算根目录或分钟store目录> [--start --end --max-symbols]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    a = ap.parse_args()
    import numpy as np
    from alphacube.engine.engine import CubeEngine
    from alphacube.label import Label
    from alphacube.metrics import cs_pearson
    from alphacube.parser import parse
    from alphacube.store import DAY_STATUS_CODES, open_cube_store

    st = open_cube_store(a.store)
    kind = type(st).__name__
    print(f"{a.store}: {kind}; {st.n_symbols} symbols x {st.n_days} days x {st.n_minutes} minutes; fields {st.fields}")
    print(f"dates {st.dates[0]}..{st.dates[-1]}; mirror: {sorted(st.meta.get('d_major_files', {})) or 'none'}")
    v = st.view(start=a.start, end=a.end, max_symbols=a.max_symbols)
    print(f"slice: {v.dates[0]}..{v.dates[-1]} {v.n_days} days x {v.n_symbols} symbols")
    status = st.read_status(v._rows(0, v.n_symbols), v.d0, v.d1)
    names = {c: n for n, c in DAY_STATUS_CODES.items()}
    for code in sorted(names):
        n = int((status == code).sum())
        if n:
            print(f"  {names[code]:18s} {n:12,d}  {n / status.size:6.2%}")
    for f in st.fields:
        cube = v.read_s(f, 0, min(v.n_symbols, 16))
        print(f"  field {f:8s} finite {np.isfinite(cube).mean():6.2%} (first 16 symbols)")
    target = Label("close", 1).compute(v)
    panel = CubeEngine(v).evaluate([parse("day_last(Close)")]).panels[0]
    ic = cs_pearson(panel, target)
    ok = int(np.isfinite(ic).sum())
    print(f"trivial factor day_last(Close) vs next-day close return: {ok}/{len(ic)} days with a finite IC")
    print("OK: store readable, mining can run" if ok else "PROBLEM: no finite IC at all -- check fields and dates")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
