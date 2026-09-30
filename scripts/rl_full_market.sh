#!/bin/bash
# 全市场 RL 挖掘(服务器上用)。 用法: scripts/rl_full_market.sh <run-id> [miner] [budget]
#   miner: alphagen(默认) | alphaqcm | alphaforge | alphasage | warmstart
#   screen -dmS <run-id> bash -c "./scripts/rl_full_market.sh <run-id> alphagen > <run-id>_run.log 2>&1"
#
# 封禁列表和 ban-archive 直接从 gp_full_market.sh 里抽出来:连跑每轮都会往那份
# 追加,这边跟着走,不另维护第二份。目标函数也和 GP 对齐 —— RL 以前的奖励是
# Pearson 池 IC,GP 从 gpu10 起用的是 |rank IC| - 0.14 x 十分位多空书日换手;
# 在 2775 个因子上量过,前者对扣费净夏普的 Spearman 只有 +0.14,后者 +0.89。
# 两条线不统一口径的话,挖出来的东西没法比,RL 还会一路挖高换手因子。
set -e
RUN_ID="${1:?用法: rl_full_market.sh <run-id> [miner] [budget]}"
MINER="${2:-alphagen}"
BUDGET="${3:-2000}"
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID 已存在,换个 run-id"; exit 2; }
export PYTHONPATH=src

# 拼成数组而不是 eval 一个字符串:封禁键里有括号和逗号,eval 只会自找麻烦
ARGS=()
while IFS= read -r key; do ARGS+=(--ban "$key"); done \
  < <(grep -oE -- '--ban "[^"]+"' scripts/gp_full_market.sh | sed 's/^--ban "//; s/"$//')
while IFS= read -r arc; do ARGS+=(--ban-archive "$arc"); done \
  < <(grep -oE -- '--ban-archive [^ \\]+' scripts/gp_full_market.sh | sed 's/^--ban-archive //')
nb=$(printf '%s\n' "${ARGS[@]}" | grep -cx -- '--ban')
na=$(printf '%s\n' "${ARGS[@]}" | grep -cx -- '--ban-archive')
echo "从 gp_full_market.sh 继承:$nb 条结构封禁 + $na 组存档"

set +e
# 不用 exec:要留下退出码(gpu14 死得无声无息就是因为没人记 $?)
PYTHONFAULTHANDLER=1 /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --mirror-root /root/autodl-tmp/AlphaCube/mirror \
  --start 20180102 --end 20251231 \
  --label open_vwap15 --horizon 1 \
  --miner "$MINER" --budget "$BUDGET" --max-expr-len 20 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.5 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  --fitness rank_ic --turnover-penalty 0.14 \
  --smooth-halflives 1,2,3,5,10,20 \
  "${ARGS[@]}" --ban-rank-corr 0.95 \
  --device cuda --seed 0 --warmup 60 \
  --progress-every 60 --panel-cache 512 --panel-cache-gb 12 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
code=$?
echo "[$RUN_ID exit $code $(date +%Y-%m-%d_%H:%M:%S)]"
exit $code
