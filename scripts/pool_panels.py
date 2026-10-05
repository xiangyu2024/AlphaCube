"""把几个已经求过值的面板目录并成一个打分目录:不重算面板、不拷面板,只写 manifest 和软链。

    cd /root/autodl-tmp/AlphaCube && python scripts/pool_panels.py \\
        --src _bench/allfac_long --src _bench/allfac_base --src _bench/allfac_hit --src _bench/allfac_hit_rl \\
        --out _bench/longpool

每个来源目录是 eval_all_factors.py 的输出:manifest.csv(列 i, run, expression)、dates.npy、panels/{i:05d}.npy。
(allfac_hit 只算了 538 个里的前 180 个,面板不在盘上的行直接不要 —— 这是预期。)

- 只收面板文件真的在盘上的行(断掉的软链算不在);
- 按表达式去重(忽略空白),来源按 --src 的顺序,同一个表达式只留最先出现的那个;
- 输出目录里:
    manifest.csv      i 从 0 起连续重编;run、expression 照抄;多一列 src = "<来源目录绝对路径>:<原来的 i>"
    panels/{i:05d}.npy  指向来源面板的软链(绝对路径),不拷贝
    dates.npy         从第一个来源拷贝,事先检查所有来源的 dates.npy 完全一样(面板的行数也要和它对得上)
- 之后的 score_all_factors.py / build_champion.py 把输出目录当成普通的打分目录用。

绝不静默覆盖:输出目录里已经有 manifest.csv 而且和这次算出来的不一样(来源多了面板、被重编号……)就报错退出,
什么都不写;一样就当没事发生(同样的命令可以重复跑,顺便确认软链都还指着对的文件);
软链的位置上已经有别的东西、或者 panels/ 里有 manifest 之外的面板,也报错退出。
注意软链的代价:来源目录里的面板之后如果被换掉 / 重编号(eval_all_factors.py 的 remap_existing 会),池子就张冠李戴 ——
再跑一遍本脚本就能发现(src 列对不上就报错)。
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FIELDS = ["i", "run", "expression", "src"]


def read_manifest(d: Path) -> List[dict]:
    path = d / "manifest.csv"
    if not path.is_file():
        raise SystemExit(f"{path} 不存在")
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        lack = {"i", "run", "expression"} - set(rd.fieldnames or [])
        if lack:
            raise SystemExit(f"{path} 缺少列 {sorted(lack)}(要 i, run, expression)")
        return list(rd)


def expr_key(expression: str) -> str:
    """去重用的键:去掉所有空白(表达式里的空白只是排版)。"""
    return "".join(expression.split())


def same_dates(a: Path, b: Path) -> bool:
    """两份 dates.npy 是不是同一串日期。先比字节;不一样再按字符串比(dtype 不同但内容相同也算一样)。"""
    if a.read_bytes() == b.read_bytes():
        return True
    try:
        x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    except ValueError:                   # 对象数组之类,不开 pickle 就没法比:当作不一样
        return False
    return x.shape == y.shape and bool((x.astype(str) == y.astype(str)).all())


def collect(srcs: Sequence[Path]) -> Tuple[List[dict], List[dict]]:
    """按来源顺序收面板。返回 (保留的行, 每个来源一行的统计);保留的行带 target(来源面板的绝对路径)。"""
    seen: Dict[str, str] = {}               # 表达式键 -> 最先收下它的来源
    kept: List[dict] = []
    stats: List[dict] = []
    for d in srcs:
        rows = read_manifest(d)
        st = dict(src=str(d), listed=len(rows), on_disk=0, missing=0, dup_same=0, dup_other=0, kept=0)
        for r in rows:
            try:
                i = int(r["i"])
            except ValueError:
                raise SystemExit(f"{d / 'manifest.csv'} 里的 i 不是整数:{r['i']!r}")
            panel = d / "panels" / f"{i:05d}.npy"
            if not panel.exists():          # exists() 跟着软链走:断掉的软链也算不在盘上
                st["missing"] += 1
                continue
            st["on_disk"] += 1
            key = expr_key(r["expression"])
            if key in seen:
                st["dup_same" if seen[key] == str(d) else "dup_other"] += 1
                continue
            seen[key] = str(d)
            st["kept"] += 1
            kept.append({"i": str(len(kept)), "run": r["run"], "expression": r["expression"],
                         "src": f"{d}:{i}", "target": str(panel)})
        stats.append(st)
    return kept, stats


def check_shapes(srcs: Sequence[Path], kept: List[dict], n_dates: int) -> None:
    """每个来源抽它的第一个保留面板看形状:行数 = 日期数,各来源的 (D, S) 一样。"""
    shapes: Dict[str, Tuple[int, ...]] = {}
    for r in kept:
        d = r["src"].rsplit(":", 1)[0]
        if d not in shapes:
            shapes[d] = tuple(np.load(r["target"], mmap_mode="r").shape)
    first_d = next(iter(shapes), None)
    for d, shp in shapes.items():
        if len(shp) != 2 or shp[0] != n_dates:
            raise SystemExit(f"{d} 的面板形状 {shp} 和 dates.npy 的 {n_dates} 天对不上")
        if shp != shapes[first_d]:
            raise SystemExit(f"面板形状不一致:{first_d} 是 {shapes[first_d]},{d} 是 {shp}")


def diff_manifest(old: List[dict], new: List[dict]) -> Optional[str]:
    """None = 一样;否则说一句第一处不同在哪。"""
    if len(old) != len(new):
        return f"行数 {len(old)} 对 {len(new)}"
    for a, b in zip(old, new):
        for k in FIELDS:
            if str(a.get(k)) != str(b[k]):
                return f"第 {b['i']} 行的 {k} 不同:现有 {a.get(k)!r},这次 {b[k]!r}"
    return None


def plan_links(panels: Path, kept: List[dict]) -> List[Tuple[Path, str]]:
    """要新建的软链 [(链接, 目标)]。位置上已经有别的东西、或有 manifest 之外的面板就报错(什么都还没写)。"""
    todo: List[Tuple[Path, str]] = []
    want = {f"{int(r['i']):05d}.npy" for r in kept}
    if os.path.islink(panels):                  # 软链到别处的 panels/:往里建链接会写进别人的目录
        raise SystemExit(f"{panels} 本身是软链(指向 {os.readlink(panels)}),不往里写;要一个真目录")
    if panels.is_dir():
        extra = sorted(n for n in os.listdir(panels) if n.endswith(".npy") and n not in want)
        if extra:
            raise SystemExit(f"{panels} 里有 {len(extra)} 个 manifest 之外的面板(如 {extra[0]}),不动它们;"
                             f"确认没用了再换个 --out 或者自己清掉")
    for r in kept:
        link = panels / f"{int(r['i']):05d}.npy"
        if os.path.islink(link):
            if os.readlink(link) != r["target"]:
                raise SystemExit(f"{link} 已经是指向别处的软链:{os.readlink(link)}(这次要指向 {r['target']})")
        elif os.path.lexists(link):
            raise SystemExit(f"{link} 已经存在,而且不是软链,不覆盖")
        else:
            todo.append((link, r["target"]))
    return todo


def print_table(stats: List[dict], total_kept: int) -> None:
    w = max([len(s["src"]) for s in stats] + [4])
    print(f"{'来源':<{w}} {'清单':>5} {'面板在盘':>8} {'缺面板':>6} {'重复(同源)':>10} {'重复(跨源)':>10} {'保留':>5}")
    for s in stats:
        print(f"{s['src']:<{w}} {s['listed']:>5} {s['on_disk']:>8} {s['missing']:>6} {s['dup_same']:>10} "
              f"{s['dup_other']:>10} {s['kept']:>5}")
    tot = {k: sum(s[k] for s in stats) for k in ("listed", "on_disk", "missing", "dup_same", "dup_other", "kept")}
    print(f"{'合计':<{w}} {tot['listed']:>5} {tot['on_disk']:>8} {tot['missing']:>6} {tot['dup_same']:>10} "
          f"{tot['dup_other']:>10} {tot['kept']:>5}")
    dups = tot["dup_same"] + tot["dup_other"]
    print(f"保留 {total_kept} 个因子;去重丢掉 {dups} 个(同一个表达式只留最先出现的来源),"
          f"面板不在盘上丢掉 {tot['missing']} 个")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", action="append", required=True, help="来源目录(eval_all_factors.py 的输出),可重复,顺序 = 去重的优先级")
    ap.add_argument("--out", required=True, help="输出目录(manifest.csv + panels/ 软链 + dates.npy)")
    a = ap.parse_args(argv)

    srcs = [Path(os.path.abspath(s)) for s in a.src]
    out = Path(os.path.abspath(a.out))
    if out in srcs:
        raise SystemExit(f"--out 不能是来源目录本身:{out}")
    for d in srcs:
        for need in ("manifest.csv", "dates.npy"):
            if not (d / need).is_file():
                raise SystemExit(f"{d / need} 不存在")
        if not (d / "panels").is_dir():
            raise SystemExit(f"{d / 'panels'} 不存在")
    for d in srcs[1:]:
        if not same_dates(srcs[0] / "dates.npy", d / "dates.npy"):
            raise SystemExit(f"{d / 'dates.npy'} 和 {srcs[0] / 'dates.npy'} 不是同一串日期,不能并成一个池")

    kept, stats = collect(srcs)
    if not kept:
        raise SystemExit("没有任何面板在盘上,不写输出")
    try:
        n_dates = int(np.load(srcs[0] / "dates.npy", allow_pickle=False).shape[0])
    except ValueError as e:
        raise SystemExit(f"{srcs[0] / 'dates.npy'} 读不出来(不开 pickle):{e}")
    check_shapes(srcs, kept, n_dates)

    man_path = out / "manifest.csv"
    if man_path.exists():
        why = diff_manifest(read_manifest(out), kept)
        if why:
            raise SystemExit(f"{man_path} 已经存在而且和这次算出来的不一样({why});不覆盖。"
                             f"来源有变化的话换个 --out,或者确认没用了自己清掉 {out}")
    if (out / "dates.npy").exists() and not same_dates(srcs[0] / "dates.npy", out / "dates.npy"):
        raise SystemExit(f"{out / 'dates.npy'} 和来源的 dates.npy 不一样,不覆盖")
    todo = plan_links(out / "panels", kept)     # 先把所有可能报错的检查做完,再动盘

    (out / "panels").mkdir(parents=True, exist_ok=True)
    for link, target in todo:
        os.symlink(target, link)
    if not (out / "dates.npy").exists():
        shutil.copyfile(srcs[0] / "dates.npy", out / "dates.npy")
    if not man_path.exists():                   # 最后写 manifest:中途断了没有 manifest,重跑就是补全
        tmp = out / "manifest.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for r in kept:
                w.writerow([r[k] for k in FIELDS])
        os.replace(tmp, man_path)
        state = f"新建 {len(todo)} 条软链,写出 {man_path}"
    else:
        state = f"和现有的一样,没有改动({len(todo)} 条软链是补上的)"

    print_table(stats, len(kept))
    print(f"-> {out}:{state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
