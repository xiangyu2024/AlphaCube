"""Every factor expression any run has archived, in one deduplicated table.

    PYTHONPATH=src python scripts/merge_archives.py [mining_runs] [out.csv]

Reads, under the runs directory (default ./mining_runs):
  <run>/checkpoints/archive_genNNN.csv   the LATEST one per run (archives are cumulative)
  <run>/members.csv                      for runs without checkpoints (early test runs)
  evals/*.json                           after-cost backtests from eval_checkpoint_costs.py

Writes one row per distinct expression (by structural key): best |IC| over runs,
signed IC / rank IC where a checkpoint carried them (every row of a gpu10+
checkpoint does: score_ic / score_rank_ic / score_turnover, with the fitness
|rank IC| - 0.14 x turnover and the family size), which runs found it, its
core (depth-1) structures, whether one of them is banned now (BANNED below),
which of its operators the current search no longer breeds (exploit-era
operators: trivial booleans, binary pow, per-element-loop kernels...), and the
after-cost numbers where computed. Sorted by best |IC|. UTF-8 with BOM so Excel
shows the Chinese headers.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphacube.catalog import default_registry
from alphacube.mining.generator import one_layer_keys
from alphacube.parser import parse

# Structures banned so far (gpu5 x6, gpu7 x3, gpu8 x3, gpu9 x6, gpu12 x12) -- keep in step with the latest run's --ban list.
BANNED = {
    "O:log(F:Amt)", "O:nt_ts_argmax(F:High)|w3", "O:cross_decay_linear(F:Amt)|w3",
    "O:day_ar1_coefficient(F:High)", "O:day_ar1_coefficient(F:Close)", "O:day_durbin_watson(F:Close)",
    "O:nt_ts_kurt(F:Vwap)|w30", "O:pow_1_5(F:Amt)", "O:day_max(F:Amt)",
    "O:day_ar1_coefficient(F:Volume)", "O:day_ar1_coefficient(F:Ret)", "O:in_cov(F:Ret,F:Close)|w30",
    "O:day_min(F:Ret)", "O:day_max(F:Ret)", "O:day_quantile_q25(F:Volume)",
    "O:nt_ts_autocorr_s1(F:Ret)|w3", "O:pow_2(F:Ret)", "O:day_kurt(F:Ret)",
    # gpu12's four blocks
    "O:day_cov(F:Amt,F:Amt)", "O:day_quantile_q25(F:Amt)", "O:cross_delta(F:Amt)|w30",
    "O:day_var(F:Ret)", "O:day_max(F:High)", "O:day_argmin(F:Amt)",
    "O:day_last(F:Volume)", "O:day_max(F:Low)", "O:day_std(F:Vwap)",
    "O:day_q_stat_std(F:Volume)", "O:cross_rank(F:High)|w241", "O:day_q_stat_std(F:Amt)",
}

runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "mining_runs")
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else runs_dir / "全部因子表达式.csv")
reg = default_registry()
searchable = {s.name for s in reg.searchable(include_boolean=False, include_resource_risk=False)}
EXCLUDED = {s.name for s in reg.list_all()} - searchable


def key_of(text):
    try:
        e = parse(text)
        return e.key(), e
    except Exception:  # noqa: BLE001 - an old run may hold an operator the parser no longer takes
        return "RAW:" + text, None


def gen_no(path):
    return int(re.search(r"gen(\d+)", os.path.basename(path)).group(1))


entries = {}


def entry(k, e, text):
    return entries.setdefault(k, {"expr": text, "e": e, "ic": 0.0, "runs": set(), "ic_test": None,
                                  "signed_ic": None, "rank_ic": None, "fit": None, "turnover": None,
                                  "dups": None, "net": None})


for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
    ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "archive_gen*.csv")), key=gen_no)
    if ckpts:
        for r in csv.DictReader(open(ckpts[-1], encoding="utf-8")):
            k, e = key_of(r["expression"])
            d = entry(k, e, r["expression"])
            # gpu10+ checkpoints: the fitness is |rank IC| - penalty x turnover, and
            # every row carries its IC as score_ic; before that the fitness WAS |IC|
            scored = bool(r.get("score_ic"))
            ic = abs(float(r["score_ic"])) if scored else float(r["fitness"])
            if ic > d["ic"]:
                d["ic"], d["expr"] = ic, r["expression"]
            d["runs"].add(run_dir.name)
            if scored:
                d["signed_ic"] = float(r["score_ic"])
                if r.get("score_rank_ic"):
                    d["rank_ic"] = float(r["score_rank_ic"])
                if r.get("score_turnover"):
                    d["turnover"] = float(r["score_turnover"])
                d["fit"] = max(float(r["fitness"]), d["fit"] if d["fit"] is not None else float("-inf"))
            elif r.get("ic_mean"):
                d["signed_ic"] = float(r["ic_mean"])
                d["rank_ic"] = float(r["rank_ic"]) if r.get("rank_ic") else None
            if r.get("dups"):
                d["dups"] = max(int(r["dups"]), d["dups"] or 0)
            if r.get("sharpe_net"):                     # after cost, traded in its IC's direction
                d["net"] = float(r["sharpe_net"])
    elif (run_dir / "members.csv").exists():
        for r in csv.DictReader(open(run_dir / "members.csv", encoding="utf-8")):
            k, e = key_of(r["expression"])
            d = entry(k, e, r["expression"])
            d["runs"].add(run_dir.name + "(members)")
            if r.get("ic_mean"):
                d["ic_test"] = abs(float(r["ic_mean"]))

net = {}
for js in sorted(glob.glob(str(runs_dir / "evals" / "*.json"))):
    for r in json.load(open(js)):
        net[key_of(r["expression"])[0]] = r

rows = []
for k, d in entries.items():
    core = sorted(one_layer_keys(d["e"])) if d["e"] is not None else []
    gone = (sorted({x.spec.name for x in d["e"].nodes() if x.spec is not None and x.spec.name in EXCLUDED})
            if d["e"] is not None else ["(无法解析)"])
    n = net.get(k)
    rows.append({
        "表达式": d["expr"],
        "最佳|IC|": round(d["ic"], 6) if d["ic"] else "",
        "IC(带符号)": "" if d["signed_ic"] is None else round(d["signed_ic"], 6),
        "rank IC": "" if d["rank_ic"] is None else round(d["rank_ic"], 6),
        "|IC|(无checkpoint的早期run)": "" if d["ic_test"] is None else round(d["ic_test"], 6),
        "出自哪些run": ",".join(sorted(d["runs"])),
        "核心结构(一层子树)": "; ".join(c[2:] if c.startswith("O:") else c for c in core),
        "适应度(|rankIC|-0.14x换手)": "" if d["fit"] is None else round(d["fit"], 6),
        "同族改写数": "" if d["dups"] is None else d["dups"],
        "现已封禁(结构)": "是" if set(core) & BANNED else "",
        "含已移出搜索的算子(漏洞写法)": "; ".join(gone),
        "夏普无成本(已翻转)": "" if n is None else round(n["sharpe"], 3),
        "夏普扣15bp": (round(n["sharpe_net"], 3) if n is not None
                       else "" if d["net"] is None else round(d["net"], 3)),
        "夏普限换手30%": "" if n is None else round(n["sharpe_capped"], 3),
        "日换手": (f"{n['turnover']:.0%}" if n is not None
                 else "" if d["turnover"] is None else f"{d['turnover']:.0%}"),
        "年化扣15bp": "" if n is None else f"{n['annualized_return_net']:+.1%}",
        "_ic": d["ic"] or (d["ic_test"] or 0.0),
    })
rows.sort(key=lambda r: -r["_ic"])
cols = [c for c in rows[0] if not c.startswith("_")]
with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
per_run = defaultdict(int)
for d in entries.values():
    for r in d["runs"]:
        per_run[r] += 1
print(f"{len(rows)} distinct expressions -> {out_path}")
print("per run:", dict(sorted(per_run.items())))
print("with after-cost backtest:", sum(1 for r in rows if r["夏普扣15bp"] != ""),
      "| using operators the search no longer breeds:", sum(1 for r in rows if r["含已移出搜索的算子(漏洞写法)"]))
