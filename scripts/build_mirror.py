"""Build D-major mirrors of a (read-only) compute root into a directory of our own.

    cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src /root/miniconda3/envs/quant/bin/python \\
      scripts/build_mirror.py /root/autodl-tmp/qmt_daily_data/fast_factor_compute mirror

The fields are (S, D, M), so a D round's read -- every symbol over a few days --
takes ~1 KB out of each 1.9 MB stride; a (D, S, M) copy makes it one contiguous
block. Every source of the compute root (the base ``minute/`` and each
``incremental/<rel>/minute/`` segment) gets its mirror at the same relative path
under the output, with a ``mirror_meta.json`` naming the sha256 of the source's
meta: a mirror is only read for the store it was built from, so a changed store
simply stops using a stale one. Runs pick mirrors up with ``--mirror-root``.

Space: the same as the fields mirrored (all 8 on the production store: ~84 GB).
A source whose mirror is already current is skipped; ``--fields`` limits the build.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("compute_root")
    ap.add_argument("out_root")
    ap.add_argument("--fields", default="", help="comma-separated; empty = every field")
    ap.add_argument("--days-per-chunk", type=int, default=64)
    a = ap.parse_args()

    from alphacube.store import MIRROR_META_FILE, SegmentedCubeStore, build_d_major_mirror

    root, out_root = Path(a.compute_root), Path(a.out_root)
    st = SegmentedCubeStore(root)
    fields = [f for f in a.fields.split(",") if f] or None
    for src in st._sources:
        rel = src.root.relative_to(root) / "minute"
        out = out_root / rel
        meta = out / MIRROR_META_FILE
        if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("source_meta_sha256") \
                == src.store.meta_sha256:
            print(f"{rel}: current, skipped", flush=True)
            continue
        t0 = time.time()
        build_d_major_mirror(src.store, fields=fields, days_per_chunk=a.days_per_chunk, out_dir=out)
        size = sum(p.stat().st_size for p in out.glob("*.dmajor.npy")) / 2**30
        print(f"{rel}: {src.store.n_days} days x {src.store.n_symbols} symbols, {size:.1f} GB "
              f"in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
