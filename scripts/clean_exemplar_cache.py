"""手工清理 exemplar 秩指纹缓存(平时不用:视图一换会自动删,超上限会自动淘汰)。

    python scripts/clean_exemplar_cache.py tmp/exemplar_cache            # 看看有什么
    python scripts/clean_exemplar_cache.py tmp/exemplar_cache --all      # 全删
    python scripts/clean_exemplar_cache.py tmp/exemplar_cache --gb 4     # 压到 4 GB
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--all", action="store_true", help="整个缓存删掉")
    ap.add_argument("--gb", type=float, default=0.0, help="压到这个大小(按最近使用时间淘汰)")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        print(f"{root} 不存在")
        return 0
    for view in sorted(p for p in root.iterdir() if p.is_dir()):
        files = list(view.glob("*.npy"))
        gb = sum(f.stat().st_size for f in files) / 1024 ** 3
        print(f"  {view.name}: {len(files)} 条 / {gb:.2f} GB")
    if a.all:
        shutil.rmtree(root, ignore_errors=True)
        print("已全部删除")
        return 0
    if a.gb > 0:
        from alphacube.mining.rankban import ExemplarCache

        for view in sorted(p for p in root.iterdir() if p.is_dir()):
            c = ExemplarCache.__new__(ExemplarCache)
            c.dir, c.max_bytes = view, int(a.gb * 1024 ** 3)
            print(f"  {view.name}: 淘汰 {c.prune()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
