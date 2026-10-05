#!/bin/bash
# 从零开始、按"纯多头"奖励做 GP 挖掘(服务器上用)。和 scripts/rl_long.sh 同一套口径:
#   用法: scripts/gp_long.sh <run-id> [budget] [end] [threads]
#   screen -dmS <run-id> bash -c "./scripts/gp_long.sh <run-id> > <run-id>_run.log 2>&1"
#
# 奖励是 --fitness long(mining/scoring.LongScorer):单独拿这个因子做前 10% 纯多头书的扣费超额夏普、
# 扣费超额年化,加上多头端(排在前一半的票)的 IC 和 rank IC。老的 --fitness rank_ic --turnover-penalty 0.14
# 原样留着(gp_full_market.sh / gp_universe.sh 还在用)。其余和 gp_universe.sh 一样:
#   - 从零开始:不继承任何结构封禁和 ban-archive,不用种子。块内的 --ban-top-n 照旧;
#   - 只挖到 end(默认 20230807,训练/测试切分点的前一天),测试段完全留出;
#   - 标签剔除"次日开盘封涨停、买不进"的收益(open_vwap15_buyable);
#   - 覆盖率门槛提到 0.8:只在一半股票上有值的因子单独拿来做多头不好用;
#   - threads / 显存封顶(GPU_FRACTION,默认 0.5):和同卡的 RL 分着用;
#   - 面板缓存 10 GB:GP 和 RL 一起跑时两边加起来要留在容器 40 GB 的匿名内存以内。
# 环境变量 UNIVERSE=<universe.npy>:只在这个股票池里挖(不设 = 全市场)。RNG_SEED:随机种子(默认 0)。
# 环境变量 FITNESS:long(第一版纯多头,默认)| long_hit(第二版:命中率 + 几种做多组合)|
#   rank_ic(对照组 —— 其余一字不改,只把奖励换回老的 |rank IC| - 0.14 x 换手)。
# 环境变量 BAN_FROM="run1 run2":这一条线的第二轮起用。把前面各轮在块尾封掉的"出现次数最多的一层子结构"
#   (日志 <run>_run.log 里的"封禁 N 个一层结构"行)和它们最后一个存档(--ban-archive,按日度秩相关 0.95 封)
#   继承过来,一上来就不许再挖这些 —— 这是去相关的办法。不设 = 真正从零开始。
#   不传递:一轮的日志里只有它自己封的,第三轮要把前面每一轮都列上(BAN_FROM="gp_long_1 gp_hit_1")。
set -e
RUN_ID="${1:?用法: gp_long.sh <run-id> [budget] [end] [threads]}"
BUDGET="${2:-1280}"
END="${3:-20230807}"
THREADS="${4:-10}"
GPU_FRACTION="${GPU_FRACTION:-0.5}"
cd /root/autodl-tmp/AlphaCube || exit 1
[ -e "mining_runs/$RUN_ID" ] && { echo "mining_runs/$RUN_ID 已存在,换个 run-id"; exit 2; }
FITNESS="${FITNESS:-long}"
case "$FITNESS" in
  long) FIT_ARGS=(--fitness long) ;;
  long_hit) FIT_ARGS=(--fitness long_hit) ;;
  rank_ic) FIT_ARGS=(--fitness rank_ic --turnover-penalty 0.14) ;;
  *) echo "FITNESS 只能是 long / long_hit / rank_ic"; exit 4 ;;
esac
BAN_ARGS=()
for prev in ${BAN_FROM:-}; do
  [ -e "${prev}_run.log" ] || { echo "BAN_FROM: 找不到 ${prev}_run.log"; exit 5; }
  arc=$(ls mining_runs/"$prev"/checkpoints/archive_gen*.csv 2>/dev/null | sort | tail -1)
  [ -n "$arc" ] || { echo "BAN_FROM: mining_runs/$prev 没有存档"; exit 5; }
  while IFS= read -r key; do [ -n "$key" ] && BAN_ARGS+=(--ban "$key"); done \
    < <(/root/miniconda3/envs/quant/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from chain_next import new_bans; print('\n'.join(new_bans(open(sys.argv[1], encoding='utf-8').read())))" "${prev}_run.log")
  BAN_ARGS+=(--ban-archive "$arc")
done
# 封禁样本的秩指纹:继承了存档就缓存到盘上(GP 和 RL、以及后面各轮共用;gp_hit_2 没开缓存,719 条算了两个多小时)。
# 用自己的目录 —— 默认的 tmp/exemplar_cache 里是全样本视图的旧缓存,换了视图会被自动删掉。
EXEM_CACHE=""
[ ${#BAN_ARGS[@]} -gt 0 ] && EXEM_CACHE="tmp/exemplar_cache_train"
UNI_ARGS=()
if [ -n "${UNIVERSE:-}" ]; then
  [ -e "$UNIVERSE" ] && [ -e "${UNIVERSE%.npy}.json" ] || { echo "股票池 $UNIVERSE 或它的 .json 不存在"; exit 3; }
  UNI_ARGS=(--universe "$UNIVERSE")
fi
export PYTHONPATH=src
export ALPHACUBE_CPU_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ALPHACUBE_GPU_FRACTION="$GPU_FRACTION"
nb=$(printf '%s\n' "${BAN_ARGS[@]}" | grep -cx -- '--ban' || true)
echo "奖励 $FITNESS,股票池 ${UNIVERSE:-全市场},挖到 $END,GP budget $BUDGET,线程 $THREADS,继承 ${BAN_FROM:-无}($nb 个子结构)"

set +e
PYTHONFAULTHANDLER=1 /root/miniconda3/envs/quant/bin/python -u -m alphacube.cli mine \
  --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
  --mirror-root /root/autodl-tmp/AlphaCube/mirror \
  --start 20180102 --end "$END" "${UNI_ARGS[@]}" \
  --label open_vwap15_buyable --horizon 1 \
  --miner gp --budget "$BUDGET" --population 64 --min-depth 3 --max-depth 5 \
  --pool mse --pool-capacity 30 --dedup 0.95 \
  --min-coverage 0.8 --min-scored-days 0.5 \
  --cost-per-side 0.0015 --turnover-cap 0.30 \
  "${FIT_ARGS[@]}" --archive-dedup 0.99 \
  --smooth-halflives 1,2,3,5,10,20 \
  --rolling-block 5 --ban-top-n 3 "${BAN_ARGS[@]}" --ban-rank-corr 0.95 --exemplar-cache "$EXEM_CACHE" \
  --device cuda --seed "${RNG_SEED:-0}" --warmup 60 \
  --checkpoint-every 1 --progress-every 60 --panel-cache 1024 --panel-cache-gb 10 \
  --combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost --combine-train-frac 0.7 \
  --run-id "$RUN_ID" --out /root/autodl-tmp/AlphaCube/mining_runs
code=$?
echo "[$RUN_ID exit $code $(date +%Y-%m-%d_%H:%M:%S)]"
exit $code
