"""一轮挖掘跑完之后,把它新封的一层结构和它的存档并进启动脚本,供下一轮使用。

`gp_chain.sh` 每轮结束调用一次。手工做过两次(gpu14->gpu15, gpu15->gpu16),动作
是固定的:日志里每个块尾的 "封禁 N 个一层结构,累计生效 M 个: A, B, C" 就是这一轮
新封的结构,它最后一个 checkpoint 就是下一轮的 --ban-archive。手工做的问题不是
难,是慢 —— gpu15 昨晚 20:36 退出,下一轮 00:17 才起,白等三个半小时。

    python scripts/chain_next.py <run>_run.log scripts/gp_full_market.sh <archive.csv>

只加不删,已经在列表里的键和存档不会重复写入;脚本原样其它内容不动。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

BAN_LINE = re.compile(r"封禁\s+\d+\s+个一层结构[,,]\s*累计生效\s+\d+\s+个[::]\s*(.+?)\s*$", re.M)
# 键之间用 ", " 分隔,而键**自己也带逗号** —— 双参数算子写成 O:day_cov(F:Low,F:Ret)。
# 按纯逗号切会把它劈成两半,gpu17 就是这么起来 90 秒就 exit 1 的
# (ValueError: ban key 'O:nt_ts_cov(F:Open' is not a depth-1 structure)。
# 键一律以 O: 开头,所以只在"逗号 + 后面跟着 O:"的地方切。
KEY_SEP = re.compile(r",\s*(?=O:)")


def new_bans(log_text: str) -> List[str]:
    """日志里所有块尾封禁行上的结构键,按出现顺序去重。"""
    keys: List[str] = []
    for match in BAN_LINE.finditer(log_text):
        for key in KEY_SEP.split(match.group(1)):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def patch_script(script: str, bans: List[str], archive: str) -> tuple:
    """返回 (新脚本, 加了哪些 ban, 加没加 archive)。"""
    have = set(re.findall(r'--ban "([^"]+)"', script))
    add = [b for b in bans if b not in have]
    if add:
        block = "".join(f'  --ban "{b}" \\\n' for b in add)
        if "  --ban-rank-corr" not in script:
            raise SystemExit("启动脚本里找不到 --ban-rank-corr 这一行,不敢乱插")
        script = script.replace("  --ban-rank-corr", block + "  --ban-rank-corr", 1)
    added_archive = False
    if archive and f"--ban-archive {archive}" not in script:
        if "  --device cuda" not in script:
            raise SystemExit("启动脚本里找不到 --device cuda 这一行,不敢乱插")
        script = script.replace("  --device cuda", f"  --ban-archive {archive} \\\n  --device cuda", 1)
        added_archive = True
    return script, add, added_archive


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    log_path, script_path, archive = Path(argv[1]), Path(argv[2]), argv[3]
    bans = new_bans(log_path.read_text(encoding="utf-8", errors="replace"))
    script, added, added_archive = patch_script(script_path.read_text(encoding="utf-8"), bans, archive)
    script_path.write_text(script, encoding="utf-8")
    total = len(re.findall(r'--ban "', script))
    print(f"并入 {len(added)} 个新封禁(累计 {total} 个): {', '.join(added) if added else '无'}")
    print(f"ban-archive {'+ ' + archive if added_archive else '已存在,未重复加入'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
