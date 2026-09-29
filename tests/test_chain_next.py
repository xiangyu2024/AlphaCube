"""连跑的交接动作:把一轮新封的结构和它的存档并进下一轮的启动脚本。

手工做过两次,内容固定;要测的是"从日志里抠对了没有"和"只加不删"。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chain_next import new_bans, patch_script      # noqa: E402

LOG = """\
     5/20  best +0.0382  archive 123  valid 31
=== 第 1 块结束(至第 5 代)===
  封禁 3 个一层结构,累计生效 39 个: O:day_autocorr_n1(F:Volume), O:day_durbin_watson(F:High), O:day_durbin_watson(F:Ret)
  按 rank 相关封禁:本块新增样本 65 个,累计 844 个
=== 第 2 块结束(至第 10 代)===
  封禁 1 个一层结构,累计生效 40 个: O:day_mean(F:Ret)
=== 第 3 块结束(至第 15 代)===
  封禁 2 个一层结构,累计生效 42 个: O:cross_delta(F:Amt)|w30, O:day_mean(F:Ret)
=== 第 4 块结束(至第 20 代)===
  封禁 3 个一层结构,累计生效 45 个: O:day_cov(F:Low,F:Ret), O:nt_ts_cov(F:Open,F:Vwap)|w5, O:day_std(F:Low)
"""

SCRIPT = """\
  --ban "O:log(F:Amt)" --ban "O:day_max(F:Amt)" \\
  --ban-rank-corr 0.95 \\
  --ban-archive mining_runs/gpu14/checkpoints/archive_gen015.csv \\
  --device cuda --seed 0 --warmup 60 \\
"""


def test_every_block_contributes_and_duplicates_collapse():
    keys = new_bans(LOG)
    assert keys == ["O:day_autocorr_n1(F:Volume)", "O:day_durbin_watson(F:High)",
                    "O:day_durbin_watson(F:Ret)", "O:day_mean(F:Ret)", "O:cross_delta(F:Amt)|w30",
                    # 双参数算子的键自己带逗号,不能按逗号切:gpu17 就死在这
                    "O:day_cov(F:Low,F:Ret)", "O:nt_ts_cov(F:Open,F:Vwap)|w5", "O:day_std(F:Low)"]


def test_patch_adds_bans_and_archive_without_touching_the_rest():
    out, added, added_archive = patch_script(SCRIPT, new_bans(LOG), "mining_runs/gpu15/checkpoints/archive_gen020.csv")
    assert len(added) == 8 and added_archive
    assert out.count('--ban "') == 10                     # 原有 2 + 新增 8
    assert '--ban "O:day_cov(F:Low,F:Ret)"' in out        # 整条,不是两半
    assert '--ban "O:log(F:Amt)"' in out                  # 只加不删
    assert out.count("--ban-archive") == 2
    assert out.index("--ban \"O:day_mean(F:Ret)\"") < out.index("--ban-rank-corr")   # 插在 rank ban 之前
    assert out.index("archive_gen020.csv") < out.index("--device cuda")
    assert out.rstrip().endswith("--device cuda --seed 0 --warmup 60 \\")


def test_second_pass_is_a_no_op():
    once, _, _ = patch_script(SCRIPT, new_bans(LOG), "mining_runs/gpu15/checkpoints/archive_gen020.csv")
    twice, added, added_archive = patch_script(once, new_bans(LOG), "mining_runs/gpu15/checkpoints/archive_gen020.csv")
    assert twice == once and added == [] and not added_archive


def test_a_log_with_no_bans_changes_nothing():
    out, added, added_archive = patch_script(SCRIPT, new_bans("     5/20 best +0.01\n"), "")
    assert out == SCRIPT and added == [] and not added_archive


def test_bad_ban_keys_are_caught_before_a_launch():
    """gpu17 是在 GPMiner 构造时才发现坏键的 —— 那时已经白起了一次。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_ban_keys import check

    good = '--ban "O:log(F:Amt)" --ban "O:day_cov(F:Low,F:Ret)" --ban "O:nt_ts_argmax(F:High)|w3"'
    assert check(good) == []
    broken = '--ban "O:nt_ts_cov(F:Open" --ban "F:Vwap)|w5" --ban "O:day_max(F:Nonexistent)"'
    bad = [k for k, _why in check(broken)]
    assert bad == ["O:nt_ts_cov(F:Open", "F:Vwap)|w5", "O:day_max(F:Nonexistent)"]
