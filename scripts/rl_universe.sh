#!/bin/bash
# 在一个股票池里从零开始做 RL 挖掘(服务器上用)。
#   用法: scripts/rl_universe.sh <run-id> [universe.npy] [miner] [budget] [end]
#   screen -dmS <run-id> bash -c "./scripts/rl_universe.sh <run-id> > <run-id>_run.log 2>&1"
#
# 和 rl_full_market.sh 的区别,其余配置(奖励、池子、平滑、合成)一致,方便和 rl1 对比:
#   - 从零开始:不继承全市场那边的结构封禁和 ban-archive(那些是全市场挖穷了才封的);
#   - 只挖到 end(默认 20230807,训练/测试切分点的前一天)。全市场的 GP/RL 生产跑一直是
#     --end 20251231,奖励看过测试段;这里把测试段完全留出来;
#   - 标签剔除"次日开盘封涨停、买不进"的收益(open_vwap15_buyable);
#   - --universe:引擎只在这段日子里进过池子的票上算,标签和每个因子面板的池外格子都置空,
#     奖励里的 IC、十分位书换手、池子合成都只看当天的成分股。
set -e
RUN_ID="${1:?用法: rl_universe.sh <run-id> [universe.npy] [miner] [budget] [end]}"
UNIVERSE="${2:-_bench/universe/hs300_proxy_v1.npy}"
MINER="${3:-alphagen}"
BUDGET="${4:-2000}"
END="${5:-20230807}"
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID 已存在,换个 run-id"; exit 2; }
[ -e "$UNIVERSE" ] && [ -e "${UNIVERSE%.npy}.json" ] || { echo "股票池 $UNIVERSE 或它的 .json 不存在"; exit 3; }
export PYTHONPATH=src
echo "股票池 $UNIVERSE,挖到 $END,miner $MINER,budget $BUDGET,不继承封禁"

set +e
PYTHONFAULTHANDLER=1 /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --mirror-root /root/autodl-tmp/AlphaCube/mirror \
  --start 20180102 --end "$END" --universe "$UNIVERSE" \
  --label open_vwap15_buyable --horizon 1 \
  --miner "$MINER" --budget "$BUDGET" --max-expr-len 20 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.5 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  --fitness rank_ic --turnover-penalty 0.14 \
  --smooth-halflives 1,2,3,5,10,20 \
  --device cuda --seed 0 --warmup 60 \
  --progress-every 60 --panel-cache 512 --panel-cache-gb 12 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
code=$?
echo "[$RUN_ID exit $code $(date +%Y-%m-%d_%H:%M:%S)]"
exit $code
