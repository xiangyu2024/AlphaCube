"""股票池挖掘结果的评估:读表达式(池子 + GP 存档前 N,去重)、训练段拟合的合成。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_universe_run as ev                               # noqa: E402


def test_run_expressions_pool_then_archive_dedup(tmp_path):
    (tmp_path / "members.csv").write_text("index,expression,weight\n0,a,1\n1,b,1\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "archive_gen001.csv").write_text("expression,fitness\nz,0.9\n", encoding="utf-8")
    (tmp_path / "checkpoints" / "archive_gen002.csv").write_text(
        "expression,fitness\nc,0.01\nb,0.05\nd,0.03\n", encoding="utf-8")
    assert ev.run_expressions(tmp_path) == ["a", "b"]
    assert ev.run_expressions(tmp_path, archive_top=2) == ["a", "b", "d"]   # 最后一个存档,按适应度,b 已在


def test_combine_train_aligns_signs_on_the_training_days():
    rng = np.random.default_rng(0)
    D, S = 100, 40
    x = rng.normal(size=(D, S))
    target = x + rng.normal(0, 0.5, (D, S))
    sig = ev.combine_train([-x, rng.normal(size=(D, S))], target, 70)     # 第一个是反号的好因子
    ic = np.mean([np.corrcoef(sig[t], target[t])[0, 1] for t in range(70, D)])
    assert ic > 0.5


def test_seed_keys_from_gp_checkpoint_and_rl_yaml(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "seeds_used.csv").write_text(
        'block,train_fitness,expression\n1,0.02,"day_std(Ret)"\n', encoding="utf-8")
    y = tmp_path / "s.yaml"
    y.write_text("seed_factors:\n- name: a\n  expression: day_mean(Ret)\n", encoding="utf-8")
    from alphacube.parser import parse
    keys = ev.seed_keys(tmp_path, [str(y)])
    assert keys == {parse("day_std(Ret)").key(), parse("day_mean(Ret)").key()}
    assert ev.seed_keys(tmp_path / "nothing") == set()
