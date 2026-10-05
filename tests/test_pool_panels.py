"""scripts/pool_panels.py:把几个面板目录并成一个打分目录 —— 只写 manifest 和软链,不拷面板;
按表达式去重(先到先得)、缺面板的行不要、dates.npy 对不上就拒绝、已有的输出不静默覆盖。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pool_panels                                               # noqa: E402

DATES = np.array(["20180102", "20180103", "20180104", "20180105"])
S = 3


def make_src(root: Path, name: str, rows, k: int, on_disk=None, dates=DATES, n_sym: int = S,
             index=None) -> Path:
    """一个来源目录。rows = [(run, expression)];index = 每行的 i(默认 0..n-1);on_disk = 面板在盘的 i 集合(默认全在)。
    面板内容是常数 1000 * k + i,这样可以从软链读回来认出它是谁的面板。"""
    d = root / name
    (d / "panels").mkdir(parents=True)
    np.save(d / "dates.npy", dates)
    index = list(range(len(rows))) if index is None else list(index)
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "run", "expression"])
        for i, (run, expr) in zip(index, rows):
            w.writerow([i, run, expr])
    for i in index:
        if on_disk is None or i in on_disk:
            np.save(d / "panels" / f"{i:05d}.npy", np.full((len(dates), n_sym), 1000.0 * k + i, dtype=np.float32))
    return d


def read(path: Path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def run(srcs, out, capsys=None):
    argv = []
    for s in srcs:
        argv += ["--src", str(s)]
    return pool_panels.main(argv + ["--out", str(out)])


def three_sources(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f(a, b)"), ("r1", "g(x)"), ("r1", "g(x)")], k=1)
    b = make_src(tmp_path, "B", [("r2", "f(a,b)"), ("r2", "h(y)"), ("r2", "k(z)")], k=2, on_disk={0, 1})
    c = make_src(tmp_path, "C", [("r3", "h(y)"), ("r3", "m(w)")], k=3)
    return a, b, c


def test_first_source_wins_dedupe_and_missing_panels(tmp_path, capsys):
    a, b, c = three_sources(tmp_path)
    out = tmp_path / "pool"
    assert run([a, b, c], out) == 0

    rows = read(out / "manifest.csv")
    assert [r["i"] for r in rows] == ["0", "1", "2", "3"]                       # 从 0 起连续重编
    assert [r["expression"] for r in rows] == ["f(a, b)", "g(x)", "h(y)", "m(w)"]   # 空白不同也是同一个表达式,留先到的文本
    assert [r["run"] for r in rows] == ["r1", "r1", "r2", "r3"]
    assert [r["src"] for r in rows] == [f"{a}:0", f"{a}:1", f"{b}:1", f"{c}:1"]     # 来源目录 + 原来的 i

    for r, (k, i) in zip(rows, [(1, 0), (1, 1), (2, 1), (3, 1)]):
        link = out / "panels" / f"{int(r['i']):05d}.npy"
        assert os.path.islink(link) and os.path.isabs(os.readlink(link))        # 软链、绝对路径,不是拷贝
        assert float(np.load(link)[0, 0]) == 1000.0 * k + i                     # 指的是对的面板
    assert sorted(os.listdir(out / "panels")) == [f"{i:05d}.npy" for i in range(4)]
    assert (out / "dates.npy").read_bytes() == (a / "dates.npy").read_bytes()
    assert not (out / "manifest.csv.tmp").exists()

    text = capsys.readouterr().out
    assert "保留 4 个因子" in text and "去重丢掉 3 个" in text and "面板不在盘上丢掉 1 个" in text
    line = {ln.split()[0]: ln.split() for ln in text.splitlines() if ln.startswith(str(tmp_path))}
    assert line[str(a)][1:] == ["3", "3", "0", "1", "0", "2"]                   # 清单 在盘 缺 同源重复 跨源重复 保留
    assert line[str(b)][1:] == ["3", "2", "1", "0", "1", "1"]
    assert line[str(c)][1:] == ["2", "2", "0", "0", "1", "1"]


def test_missing_panel_is_not_counted_as_the_first_seen_copy(tmp_path):
    # 先出现的那份没有面板,后面有面板的那份不能被当成重复丢掉
    a = make_src(tmp_path, "A", [("r1", "f(x)"), ("r1", "g(x)")], k=1, on_disk={1})
    b = make_src(tmp_path, "B", [("r2", "f(x)")], k=2)
    out = tmp_path / "pool"
    run([a, b], out)
    rows = read(out / "manifest.csv")
    assert [(r["run"], r["expression"]) for r in rows] == [("r1", "g(x)"), ("r2", "f(x)")]


def test_source_indices_need_not_be_consecutive(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f"), ("r1", "g"), ("r1", "h")], k=1, index=[5, 9, 12], on_disk={5, 12})
    out = tmp_path / "pool"
    run([a], out)
    rows = read(out / "manifest.csv")
    assert [(r["i"], r["expression"], r["src"]) for r in rows] == [("0", "f", f"{a}:5"), ("1", "h", f"{a}:12")]
    assert float(np.load(out / "panels" / "00001.npy")[0, 0]) == 1000.0 + 12


def test_rerun_is_a_noop_but_a_changed_pool_is_refused(tmp_path, capsys):
    a, b, c = three_sources(tmp_path)
    out = tmp_path / "pool"
    run([a, b, c], out)
    before = (out / "manifest.csv").read_bytes()
    assert run([a, b, c], out) == 0                                             # 同样的命令重复跑:没事
    assert (out / "manifest.csv").read_bytes() == before
    assert "和现有的一样" in capsys.readouterr().out

    np.save(b / "panels" / "00002.npy", np.zeros((len(DATES), S), dtype=np.float32))   # 来源多了一个面板
    with pytest.raises(SystemExit) as e:
        run([a, b, c], out)
    assert "不覆盖" in str(e.value)
    assert (out / "manifest.csv").read_bytes() == before                        # 什么都没动
    assert sorted(os.listdir(out / "panels")) == [f"{i:05d}.npy" for i in range(4)]


def test_rerun_restores_links_after_an_interrupted_run(tmp_path):
    a, b, c = three_sources(tmp_path)
    out = tmp_path / "pool"
    run([a, b, c], out)
    os.unlink(out / "panels" / "00002.npy")                                     # 中途断掉:少一条软链
    run([a, b, c], out)
    assert float(np.load(out / "panels" / "00002.npy")[0, 0]) == 2000.0 + 1

    # 没有 manifest(最后才写)的半成品目录:重跑就是补全
    os.unlink(out / "manifest.csv")
    os.unlink(out / "panels" / "00003.npy")
    run([a, b, c], out)
    assert len(read(out / "manifest.csv")) == 4 and os.path.islink(out / "panels" / "00003.npy")


def test_dates_must_be_identical(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1)
    b = make_src(tmp_path, "B", [("r2", "g")], k=2, dates=np.array(["20180102", "20180103", "20180104", "20180108"]))
    out = tmp_path / "pool"
    with pytest.raises(SystemExit) as e:
        run([a, b], out)
    assert "不是同一串日期" in str(e.value)
    assert not out.exists()                                                     # 拒绝之前什么都没写


def test_dates_with_another_dtype_but_same_values_are_fine(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1, dates=np.array([20180102, 20180103, 20180104, 20180105]))
    b = make_src(tmp_path, "B", [("r2", "g")], k=2, dates=np.array(["20180102", "20180103", "20180104", "20180105"]))
    assert run([a, b], tmp_path / "pool") == 0


def test_panel_shapes_must_match(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1)
    b = make_src(tmp_path, "B", [("r2", "g")], k=2, n_sym=S + 1)
    with pytest.raises(SystemExit) as e:
        run([a, b], tmp_path / "pool")
    assert "形状不一致" in str(e.value) and not (tmp_path / "pool").exists()

    c = make_src(tmp_path, "C", [("r3", "h")], k=3)
    np.save(c / "panels" / "00000.npy", np.zeros((len(DATES) + 1, S), dtype=np.float32))   # 行数和 dates 对不上
    with pytest.raises(SystemExit) as e:
        run([c], tmp_path / "pool2")
    assert "对不上" in str(e.value)


def test_foreign_files_in_the_output_are_never_overwritten(tmp_path):
    a, b, c = three_sources(tmp_path)
    out = tmp_path / "pool"
    (out / "panels").mkdir(parents=True)
    mine = out / "panels" / "00001.npy"
    mine.write_bytes(b"not a link")
    with pytest.raises(SystemExit) as e:
        run([a, b, c], out)
    assert "不是软链" in str(e.value)
    assert mine.read_bytes() == b"not a link"
    assert not (out / "manifest.csv").exists() and not os.path.lexists(out / "panels" / "00000.npy")   # 先查后写

    mine.unlink()
    os.symlink(str(a / "panels" / "00000.npy"), out / "panels" / "00001.npy")           # 指向别处的软链
    with pytest.raises(SystemExit) as e:
        run([a, b, c], out)
    assert "指向别处" in str(e.value)

    os.unlink(out / "panels" / "00001.npy")
    (out / "panels" / "00007.npy").write_bytes(b"x")                                    # manifest 之外的面板
    with pytest.raises(SystemExit) as e:
        run([a, b, c], out)
    assert "manifest 之外" in str(e.value)
    assert not (out / "manifest.csv").exists()


def test_panels_dir_that_is_a_symlink_is_refused(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1)
    out = tmp_path / "pool"
    out.mkdir()
    os.symlink(str(a / "panels"), out / "panels")                                       # 指向来源的 panels/:往里写就写进来源了
    before = sorted(os.listdir(a / "panels"))
    with pytest.raises(SystemExit) as e:
        run([a], out)
    assert "本身是软链" in str(e.value)
    assert sorted(os.listdir(a / "panels")) == before and not (out / "manifest.csv").exists()


def test_existing_dates_in_output_must_match(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1)
    out = tmp_path / "pool"
    out.mkdir()
    np.save(out / "dates.npy", np.array(["19990101"]))
    with pytest.raises(SystemExit) as e:
        run([a], out)
    assert "dates.npy 和来源的" in str(e.value) or "不一样" in str(e.value)
    assert not (out / "manifest.csv").exists()


def test_relative_paths_give_absolute_links_and_the_same_manifest(tmp_path, monkeypatch):
    a, b, c = three_sources(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert pool_panels.main(["--src", "A", "--src", "B", "--src", "C", "--out", "pool"]) == 0
    assert os.path.isabs(os.readlink(tmp_path / "pool" / "panels" / "00000.npy"))
    rows = read(tmp_path / "pool" / "manifest.csv")
    assert all(os.path.isabs(r["src"]) for r in rows)
    monkeypatch.chdir("/")                                                             # 换个目录、改用绝对路径:同一个池
    assert run([a, b, c], tmp_path / "pool") == 0


def test_bad_inputs_are_refused_without_writing(tmp_path):
    a = make_src(tmp_path, "A", [("r1", "f")], k=1, on_disk=set())
    with pytest.raises(SystemExit) as e:                                               # 一个面板都不在盘上
        run([a], tmp_path / "pool")
    assert "没有任何面板" in str(e.value) and not (tmp_path / "pool").exists()
    with pytest.raises(SystemExit):                                                    # --out 就是来源
        run([a], a)
    with pytest.raises(SystemExit):                                                    # 来源目录不存在
        run([tmp_path / "nope"], tmp_path / "pool")
    d = make_src(tmp_path, "D", [("r", "f")], k=4)
    (d / "manifest.csv").write_text("i,expression\n0,f\n", encoding="utf-8")           # 缺 run 列
    with pytest.raises(SystemExit) as e:
        run([d], tmp_path / "pool")
    assert "缺少列" in str(e.value)
