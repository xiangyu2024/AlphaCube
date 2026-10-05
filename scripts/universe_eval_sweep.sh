#!/bin/bash
# 股票池挖掘的一个 run:全区间干净评估(eval_universe_run.py)+ 每个合成信号一次 72 格真实回测
# (和 docs/可交易回测.md §13.2 同一口径;服务器上用):
#   ./scripts/universe_eval_sweep.sh <run-id> [--mark-seeds seeds.yaml ...]
set -u
cd /root/autodl-tmp/AlphaCube
export PYTHONPATH=src PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/quant/bin/python
U=_bench/universe/hs300_proxy_v1.npy
T=_bench/tradable
RUN=${1:?用法: universe_eval_sweep.sh <run-id> [eval_universe_run 的其他参数]}
shift
LOG=_bench/universe/eval_$RUN.log
GRID="--pool 1.0 --caps1 0.01,0.02,0.03,0.05 --keeps 0,0.3,0.5 --bands 0,0.5 --halflives 5,10,20 --stage1-only --workers 8 --top 15"
$PY -u scripts/eval_universe_run.py --run mining_runs/$RUN --universe $U --out-dir _bench/universe "$@" > $LOG 2>&1
echo "[eval exit $?]" >> $LOG
for SIG in _bench/universe/signal_${RUN}_*.npy; do
  M=$(basename $SIG .npy); M=${M#signal_${RUN}_}
  $PY -u scripts/tradable_backtest.py sweep --inputs $T/market_20180102_20251231.npz --signal $SIG --cost retail \
      --universe $U $GRID --out $T/sweep_${RUN}_$M.json > $T/sweep_${RUN}_$M.log 2>&1
  echo "[sweep $M exit $?]" >> $LOG
done
echo "[all done]" >> $LOG
