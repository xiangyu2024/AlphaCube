"""校验启动脚本里每一个 --ban 键,在合并之后、起下一轮之前跑。

gpu17 起来 90 秒就 exit 1:交接脚本把双参数算子的键 `O:nt_ts_cov(F:Open,F:Vwap)|w5`
按逗号切成了两半,GPMiner 构造时才发现。一个坏键的代价不该是"浪费一次启动 +
一次人工排查" —— 这里几秒钟验完,连跑在合并之后立刻调用。

    python scripts/check_ban_keys.py scripts/gp_full_market.sh
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphacube.catalog import default_registry          # noqa: E402
from alphacube.mining.gp import check_ban_key           # noqa: E402


def check(script_text: str) -> list:
    """返回坏键列表 [(键, 原因), ...];全对时是空的。"""
    registry = default_registry()
    fields = list(registry.fields)
    bad = []
    for key in re.findall(r'--ban "([^"]+)"', script_text):
        try:
            check_ban_key(key, registry, fields)
        except ValueError as exc:
            bad.append((key, str(exc)))
    return bad


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    text = Path(argv[1]).read_text(encoding="utf-8")
    bad = check(text)
    total = len(re.findall(r'--ban "', text))
    for key, why in bad:
        print(f"坏键 {key!r}: {why}")
    print(f"{total} 个封禁键,{len(bad)} 个不合法")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
