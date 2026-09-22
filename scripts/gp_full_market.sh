#!/bin/bash
# Full-market GP run (on the server).  Usage: scripts/gp_full_market.sh <run-id>
# Launch it detached, e.g.:
#   screen -dmS <run-id> bash -c "./scripts/gp_full_market.sh <run-id> > <run-id>_run.log 2>&1"
#   screen -dmS <run-id>rep bash -c "./scripts/gp_report_loop.sh <run-id>"
#
# 20 generations = 4 rolling blocks x 5; pool capped at 30 (block reports score the top 120,
# the final commit the top 60) -- memory-bounded; uncapped pools died on the 90 GiB container.
# Starts with every structure earlier runs banned (30: gpu5..gpu9 and gpu12's four blocks) plus the
# rank ban: a candidate whose daily ranks correlate >= 0.95 with a factor of a banned family is refused
# (exemplars from the five archives below, evaluated 150 per engine call). Costs 15 bp/side, 30% turnover cap in every report.
# From gpu10 on (user's decisions of 2026-09-19):
#   fitness = |rank IC| - 0.14 x daily turnover of the decile book (0.14 = 15 bp/side in rank-IC units);
#   archive families: trees whose daily ranks correlate >= 0.99 are one factor -- the archive keeps
#     the best, and within a generation only that one breeds;
#   every combination is also traded through an EMA whose half-life is picked on the training days.
set -e
RUN_ID="$1"
[ -n "$RUN_ID" ] || { echo "usage: $0 <run-id>"; exit 2; }
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID exists; pick a new run-id"; exit 2; }
export PYTHONPATH=src
exec /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --start 20180102 --end 20251231 \
  --label open_vwap15 --horizon 1 \
  --miner gp --budget 1280 --population 64 --min-depth 2 --max-depth 5 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.5 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  --fitness rank_ic --turnover-penalty 0.14 --archive-dedup 0.99 \
  --smooth-halflives 1,2,3,5,10,20 \
  --rolling-block 5 --ban-top-n 3 \
  --ban "O:log(F:Amt)" --ban "O:nt_ts_argmax(F:High)|w3" --ban "O:cross_decay_linear(F:Amt)|w3" \
  --ban "O:day_ar1_coefficient(F:High)" --ban "O:day_ar1_coefficient(F:Close)" --ban "O:day_durbin_watson(F:Close)" \
  --ban "O:nt_ts_kurt(F:Vwap)|w30" --ban "O:pow_1_5(F:Amt)" --ban "O:day_max(F:Amt)" \
  --ban "O:day_ar1_coefficient(F:Volume)" --ban "O:day_ar1_coefficient(F:Ret)" --ban "O:in_cov(F:Ret,F:Close)|w30" \
  --ban "O:day_min(F:Ret)" --ban "O:day_max(F:Ret)" --ban "O:day_quantile_q25(F:Volume)" \
  --ban "O:nt_ts_autocorr_s1(F:Ret)|w3" --ban "O:pow_2(F:Ret)" --ban "O:day_kurt(F:Ret)" \
  --ban "O:day_cov(F:Amt,F:Amt)" --ban "O:day_quantile_q25(F:Amt)" --ban "O:cross_delta(F:Amt)|w30" \
  --ban "O:day_var(F:Ret)" --ban "O:day_max(F:High)" --ban "O:day_argmin(F:Amt)" \
  --ban "O:day_last(F:Volume)" --ban "O:day_max(F:Low)" --ban "O:day_std(F:Vwap)" \
  --ban "O:day_q_stat_std(F:Volume)" --ban "O:cross_rank(F:High)|w241" --ban "O:day_q_stat_std(F:Amt)" \
  --ban-rank-corr 0.95 \
  --ban-archive mining_runs/gpu5/checkpoints/archive_gen010.csv \
  --ban-archive mining_runs/gpu7/checkpoints/archive_gen007.csv \
  --ban-archive mining_runs/gpu8/checkpoints/archive_gen008.csv \
  --ban-archive mining_runs/gpu9/checkpoints/archive_gen010.csv \
  --ban-archive mining_runs/gpu12/checkpoints/archive_gen020.csv \
  --device cuda --seed 0 --warmup 60 \
  --checkpoint-every 1 --progress-every 60 --panel-cache 1024 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
