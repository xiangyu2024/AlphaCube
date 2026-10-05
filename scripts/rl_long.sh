#!/bin/bash
# 从零开始、按"纯多头"奖励做 RL 挖掘(服务器上用)。和 scripts/gp_long.sh 同一套口径:
#   用法: scripts/rl_long.sh <run-id> [miner] [budget] [end]
#   screen -dmS <run-id> bash -c "./scripts/rl_long.sh <run-id> > <run-id>_run.log 2>&1"
#
# 奖励是 --fitness long(mining/scoring.LongScorer):候选自己先过纯多头口径的门槛,进了池以后的奖励是
# 池子合成信号的纯多头适应度(做多它最高的 10%)。老的奖励原样留着(rl_full_market.sh / rl_universe.sh)。
# 其余和 rl_universe.sh 一样:从零开始(不继承封禁、不用种子)、只挖到 end(默认 20230807)、
# 标签是 open_vwap15_buyable;覆盖率门槛 0.8;面板缓存 6 GB(和 GP 一起跑时留在 40 GB 以内)。
# 环境变量 UNIVERSE=<universe.npy>:只在这个股票池里挖(不设 = 全市场)。RNG_SEED:随机种子(默认 0)。
# 环境变量 FITNESS:long(默认)| long_hit(第二版:命中率 + 几种做多组合)。
# 环境变量 MIN_SOLO(默认 0):进池门槛 —— 候选自己的适应度(已经扣了换手)低于它就不进池,奖励就是它自己的适应度。
#   rl_long_1 没有这道门槛,池子只按合成信号的 IC 收人,30 个成员全是高换手因子。MIN_SOLO=off 关掉。
# 环境变量 BAN_FROM="run1 run2":继承这些 GP run 在块尾封掉的出现次数最多的子结构和它们的存档(见 gp_long.sh)。
set -e
RUN_ID="${1:?用法: rl_long.sh <run-id> [miner] [budget] [end]}"
MINER="${2:-alphagen}"
BUDGET="${3:-2000}"
END="${4:-20230807}"
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID 已存在,换个 run-id"; exit 2; }
FITNESS="${FITNESS:-long}"
case "$FITNESS" in long|long_hit) ;; *) echo "FITNESS 只能是 long / long_hit"; exit 4 ;; esac
SOLO_ARGS=("--rl-min-solo=${MIN_SOLO:-0}")
[ "${MIN_SOLO:-0}" = off ] && SOLO_ARGS=()
BAN_ARGS=()
for prev in ${BAN_FROM:-}; do
  [ -e "${prev}_run.log" ] || { echo "BAN_FROM: 找不到 ${prev}_run.log"; exit 5; }
  arc=$(ls mining_runs/"$prev"/checkpoints/archive_gen*.csv 2>/dev/null | sort | tail -1)
  [ -n "$arc" ] || { echo "BAN_FROM: mining_runs/$prev 没有存档"; exit 5; }
  while IFS= read -r key; do [ -n "$key" ] && BAN_ARGS+=(--ban "$key"); done \
    < <(/root/miniconda3/envs/quant/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from chain_next import new_bans; print('\n'.join(new_bans(open(sys.argv[1], encoding='utf-8').read())))" "${prev}_run.log")
  BAN_ARGS+=(--ban-archive "$arc")
done
# 封禁样本的秩指纹缓存和 gp_long.sh 用同一个目录:视图一样,GP 先算过的 RL 直接读
[ ${#BAN_ARGS[@]} -gt 0 ] && BAN_ARGS+=(--ban-rank-corr 0.95 --exemplar-cache tmp/exemplar_cache_train)
UNI_ARGS=()
if [ -n "${UNIVERSE:-}" ]; then
  [ -e "$UNIVERSE" ] && [ -e "${UNIVERSE%.npy}.json" ] || { echo "股票池 $UNIVERSE 或它的 .json 不存在"; exit 3; }
  UNI_ARGS=(--universe "$UNIVERSE")
fi
export PYTHONPATH=src
THREADS="${THREADS:-10}"
export ALPHACUBE_CPU_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ALPHACUBE_GPU_FRACTION="${GPU_FRACTION:-0.3}"
echo "奖励 $FITNESS,股票池 ${UNIVERSE:-全市场},挖到 $END,miner $MINER,budget $BUDGET,线程 $THREADS,进池门槛 ${MIN_SOLO:-0},继承 ${BAN_FROM:-无}"

set +e
PYTHONFAULTHANDLER=1 /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --mirror-root /root/autodl-tmp/AlphaCube/mirror \
  --start 20180102 --end "$END" "${UNI_ARGS[@]}" \
  --label open_vwap15_buyable --horizon 1 \
  --miner "$MINER" --budget "$BUDGET" --max-expr-len 20 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.8 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  --fitness "$FITNESS" "${SOLO_ARGS[@]}" "${BAN_ARGS[@]}" \
  --smooth-halflives 1,2,3,5,10,20 \
  --device cuda --seed "${RNG_SEED:-0}" --warmup 60 \
  --progress-every 60 --panel-cache 512 --panel-cache-gb 6 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
code=$?
echo "[$RUN_ID exit $code $(date +%Y-%m-%d_%H:%M:%S)]"
exit $code
