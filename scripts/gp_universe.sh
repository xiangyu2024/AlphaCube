#!/bin/bash
# 在一个股票池里从零开始做 GP 挖掘(服务器上用)。和 scripts/rl_universe.sh 同一套口径:
#   用法: scripts/gp_universe.sh <run-id> [universe.npy] [budget] [end] [threads]
#   screen -dmS <run-id> bash -c "./scripts/gp_universe.sh <run-id> > <run-id>_run.log 2>&1"
#
# 和 gp_full_market.sh 的区别(GP 自己的参数 —— 种群、深度、分块、族去重、块内结构封禁 —— 不变):
#   - 从零开始:不继承全市场那边的结构封禁和 ban-archive。块内的 --ban-top-n 照旧(那是这一轮自己挖穷了才封);
#   - 只挖到 end(默认 20230807,训练/测试切分点的前一天),测试段完全留出;
#   - 标签剔除"次日开盘封涨停、买不进"的收益(open_vwap15_buyable);
#   - --universe:引擎只在这段日子里进过池子的票上算,标签和每个因子面板的池外格子都置空;
#   - threads:CPU 线程数封顶(默认 10)。容器配额 25 核,同时还有别的挖掘在跑时不封顶会互相抢;
#   - 显存封顶到整卡的 GPU_FRACTION(默认 0.5),和同卡的 RL 分着用;
#   - 不用指纹磁盘缓存(--exemplar-cache ""):这一轮没有 ban-archive,块内封禁的样本本来就在内存里。
set -e
RUN_ID="${1:?用法: gp_universe.sh <run-id> [universe.npy] [budget] [end] [threads]}"
UNIVERSE="${2:-_bench/universe/hs300_proxy_v1.npy}"
BUDGET="${3:-1280}"
END="${4:-20230807}"
THREADS="${5:-10}"
GPU_FRACTION="${GPU_FRACTION:-0.5}"
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID 已存在,换个 run-id"; exit 2; }
[ -e "$UNIVERSE" ] && [ -e "${UNIVERSE%.npy}.json" ] || { echo "股票池 $UNIVERSE 或它的 .json 不存在"; exit 3; }
export PYTHONPATH=src
export ALPHACUBE_CPU_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ALPHACUBE_GPU_FRACTION="$GPU_FRACTION"
echo "股票池 $UNIVERSE,挖到 $END,GP budget $BUDGET,线程 $THREADS,不继承封禁"

set +e
PYTHONFAULTHANDLER=1 /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --mirror-root /root/autodl-tmp/AlphaCube/mirror \
  --start 20180102 --end "$END" --universe "$UNIVERSE" \
  --label open_vwap15_buyable --horizon 1 \
  --miner gp --budget "$BUDGET" --population 64 --min-depth 3 --max-depth 5 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.5 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  --fitness rank_ic --turnover-penalty 0.14 --archive-dedup 0.99 \
  --smooth-halflives 1,2,3,5,10,20 \
  --rolling-block 5 --ban-top-n 3 --ban-rank-corr 0.95 --exemplar-cache "" \
  --device cuda --seed 0 --warmup 60 \
  --checkpoint-every 1 --progress-every 60 --panel-cache 1024 --panel-cache-gb 16 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
code=$?
echo "[$RUN_ID exit $code $(date +%Y-%m-%d_%H:%M:%S)]"
exit $code
