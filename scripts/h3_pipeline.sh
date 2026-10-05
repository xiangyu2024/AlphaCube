#!/bin/bash
# H3 的整条流水线(服务器上用,每步退出码写进日志,非 0 就停):
#   纯多头奖励挖出来的因子 -> 并成一个池 -> 打分 -> 聚类 + 两层合成成一个信号 -> 真实纯多头回测 36 格;
#   再和现有冠军等权混合、同一套回测;最后把四个信号的 sweep 并排成一张表。
#
# H3 = 用纯多头奖励挖出来的因子,合成一个信号以后,在真实纯多头回测(100 万、A 股实际费率、整手、上市不足 60 天不买、
#   配置在训练段选、测试段只报数)里好不好用;和现有冠军比(测试段超额夏普 +2.28,换手修复后 +2.38),以及和它混合以后比。
#   注意池子里有 gp_base_1(老奖励的对照组,约 40%):它是"这条研究线挖出来的全部因子",不全是纯多头奖励挖的。
#   只要纯多头奖励的那几个:POOL_SRCS="_bench/allfac_long _bench/allfac_hit _bench/allfac_hit_rl" TAG=longonly。
#
# 启动(screen 里跑,退出码由脚本自己写在日志末尾 "[h3_pipeline exit N 时间]"):
#   screen -dmS h3 bash -c "./scripts/h3_pipeline.sh > _bench/long_reward/h3_pipeline.log 2>&1"
# 预演(不算任何东西:查输入在不在、输出会不会覆盖、内存门槛,然后原样打印每一步的命令):
#   DRY_RUN=1 ./scripts/h3_pipeline.sh
#
# 步骤(FROM / TO 用字母选一段,默认 a 到 f;失败后修好了用 FROM=c 之类接着跑):
#   a 并池   scripts/pool_panels.py -> _bench/$TAG(只写 manifest 和软链,几秒)
#   b 打分   score_all_factors.py --buyable --no-rows(和 _bench/trainto/{legacy,allfac_buyable} 同一组参数;写 turnover_train)
#   c 合成   build_champion.py --buyable(和 _bench/trainto/legacy_signal.npy 同一组参数)-> _bench/${TAG}_signal.npy(+ .families.*)
#   d 回测   tradable_backtest.py sweep,36 格全市场(和冠军、手写库同一套网格和费率)-> _bench/tradable/sweep_full36_${TAG}_signal.json
#   e 混合   blend_signals.py 与 $CHAMPION 等权 -> _bench/blend_${TAG}_champion.npy,同一套 36 格回测
#   f 汇总   sweep_table.py:新池子、混合、现冠军、手写库(+ 旧冠军 +2.28 的出处)并排
#
# 预计(参考 _bench/trainto/pipeline.log 和 _bench/legacy/pipeline.log 的实测;池子 1168 个面板):
#   b 打分     参考 0.36 s/因子(8 进程 x 3 线程 = 24 核:挖掘 4206 个 1511 s、手写 1392 个 510 s);
#              本脚本 4 进程 x 3 线程 = 12 核,约 0.7 s/因子 -> 1168 个约 14 分钟
#   c 合成     参考 17~20 分钟(手写库 1020 s、挖掘 1202 s 写出信号,另加 ~50 s 的对照表);过门槛的族不到 100 个会更快
#   d、e 回测  每份 36 格总共约 12~13 s(含加载行情;参考日志里 [11s] 和 pipeline.log 的时间戳);blend 约 1 分钟
#   合计       约 30~40 分钟
# 内存(匿名;参考日志里没有实测,下面是按代码里数组大小估的,1 个 (天 x 股票) 面板 = 81 MB float64 / 40 MB float32):
#   b ~8~10 GB   4 个打分进程各 ~2 GB(量级和 long_report.py 进程的 1.4~1.6 GB RSS 相当)+ 主进程算标签
#   c 峰值 ~20 GB  100 个族时最后一层 lstsq:100 个族因子 float64 8 GB + 取训练段的列表和堆叠各 ~5.7 GB;
#                  族内一步(60 个成员)约 12 GB;族数不到 100 个就按比例少
#   d、e 回测 ~4~6 GB  行情和信号 ~1 GB + 12 个 fork 出来的子进程
#   所以每一步前面有个内存门槛(MEM_GATE=1):容器匿名内存 + 这一步的预计 > ANON_LIMIT_GB(40)就等,不会把别人的任务挤掉。
# 核数:ALPHACUBE_CPU_THREADS / OMP / MKL / OPENBLAS 全部封到 THREADS(12);打分的每个进程 3 线程(= 参考的 25//8),
#   回测的 12 个进程每个 1 线程;CUDA_VISIBLE_DEVICES 置空,这条流水线不碰 GPU。
#
# 环境变量(都可以改):
#   TAG(longpool)  POOL_SRCS  CHAMPION(_bench/trainto/champion_signal_buyable.npy)  MKT  PY(quant 环境的 python)
#   THREADS(12)  SCORE_WORKERS(THREADS/3)  SWEEP_WORKERS(THREADS)  NICE(10)
#   FROM(a)  TO(f)  DRY_RUN(0)  OVERWRITE(0:要写的输出已经存在就拒绝;1:放行)
#   MEM_GATE(1)  ANON_LIMIT_GB(40)  MEM_WAIT_MAX_MIN(720)  MEM_POLL_SEC(60)  MEMSTAT(/sys/fs/cgroup/memory.stat)
#   MIN_FREE_GB(15)  PROJ(/root/autodl-tmp/AlphaCube)
#   STORE:只用来预检(score_all_factors / build_champion 的 --store 默认值就是它,这里不往下传)
set -u

PROJ=${PROJ:-/root/autodl-tmp/AlphaCube}
PY=${PY:-/root/miniconda3/envs/quant/bin/python}
STORE=${STORE:-/root/autodl-tmp/qmt_daily_data/fast_factor_compute}
TAG=${TAG:-longpool}
POOL_SRCS=${POOL_SRCS:-"_bench/allfac_long _bench/allfac_base _bench/allfac_hit _bench/allfac_hit_rl"}
CHAMPION=${CHAMPION:-_bench/trainto/champion_signal_buyable.npy}
MKT=${MKT:-_bench/tradable/market_20180102_20251231.npz}
DRY_RUN=${DRY_RUN:-0}
FROM=${FROM:-a}
TO=${TO:-f}
OVERWRITE=${OVERWRITE:-0}
THREADS=${THREADS:-12}
NICE=${NICE:-10}
SCORE_WORKERS=${SCORE_WORKERS:-$(( THREADS / 3 > 0 ? THREADS / 3 : 1 ))}
SWEEP_WORKERS=${SWEEP_WORKERS:-$THREADS}
MEM_GATE=${MEM_GATE:-1}
ANON_LIMIT_GB=${ANON_LIMIT_GB:-40}
MEM_WAIT_MAX_MIN=${MEM_WAIT_MAX_MIN:-720}
MEM_POLL_SEC=${MEM_POLL_SEC:-60}
MEMSTAT=${MEMSTAT:-/sys/fs/cgroup/memory.stat}
MIN_FREE_GB=${MIN_FREE_GB:-15}

P=_bench/$TAG                                       # 池子目录:manifest + 面板软链 + 打分结果
T=_bench/tradable
SIG=_bench/${TAG}_signal.npy
FAM=_bench/${TAG}_signal.families                   # build_champion 在信号旁边写 .families.npy / .families.json
BLEND=_bench/blend_${TAG}_champion.npy
SW_POOL=$T/sweep_full36_${TAG}_signal.json
SW_BLEND=$T/sweep_full36_blend_${TAG}_champion.json
# 对照:都是 _bench/trainto/pipeline.sh 的 G36 网格,训练段换手修复之后(turnover_train)的口径
REF_CHAMP=_bench/trainto/sweep_mined_full_G36_new.json      # 信号 _bench/trainto/champion_signal_buyable.npy
REF_LEGACY=_bench/trainto/sweep_legacy_full_G36_new.json    # 信号 _bench/trainto/legacy_signal.npy
REF_CHAMP_OLD=_bench/trainto/sweep_mined_full_G36_old.json  # 修复前的冠军(= 一直说的 +2.28)
# 36 格网格和费率:和 _bench/trainto/pipeline.sh 的 G36、scripts/legacy_pipeline.sh 第 4 步逐字相同(--workers 在下面单独传,不影响结果)
G36="--caps1 0.005,0.01 --halflives 5,10 --bands 0,0.5,1.0 --keeps 0,0.3,0.5 --stage1-only --top 15"

ts() { date '+%F %T'; }
num() { case $1 in a) echo 1;; b) echo 2;; c) echo 3;; d) echo 4;; e) echo 5;; f) echo 6;; *) echo 0;; esac; }
FROM_N=$(num "$FROM"); TO_N=$(num "$TO")
if [ "$FROM_N" -lt 1 ] || [ "$TO_N" -lt 1 ] || [ "$FROM_N" -gt "$TO_N" ]; then
  echo "FROM / TO 要是 a~f 之一且 FROM 不晚于 TO(a 并池 b 打分 c 合成 d 池子回测 e 混合+回测 f 汇总):FROM=$FROM TO=$TO"
  exit 2
fi
want() { local n; n=$(num "$1"); [ "$n" -ge "$FROM_N" ] && [ "$n" -le "$TO_N" ]; }

cd "$PROJ" || { echo "进不了 $PROJ"; exit 1; }
export PYTHONPATH=src PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export ALPHACUBE_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
ENV_LINE="PYTHONPATH=src PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= ALPHACUBE_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS"

LOCK=_bench/.h3_${TAG}.lock
LOCKED=0
finish() {
  local rc=$?
  [ "$LOCKED" = 1 ] && rmdir "$LOCK" 2>/dev/null
  if [ "$DRY_RUN" = 1 ]; then echo "[dry-run exit $rc]"; else echo "[h3_pipeline exit $rc $(date +%F_%T)]"; fi
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

# ---------------------------------------------------------------- 预检(只读)
miss=0
chk_x()    { [ -x "$1" ] || { echo "  缺(不可执行): $1"; miss=1; }; }
chk_file() { [ -s "$1" ] || { echo "  缺(或为空): $1"; miss=1; }; }
chk_dir()  { [ -d "$1" ] || { echo "  缺目录: $1"; miss=1; }; }
chk_new() {   # 这一步要新写的输出:已经存在就拒绝(OVERWRITE=1 放行),绝不静默覆盖
  { [ -e "$1" ] || [ -L "$1" ]; } || return 0
  if [ "$OVERWRITE" = 1 ]; then echo "  (OVERWRITE=1)会覆盖: $1"; else echo "  已存在,拒绝覆盖(要重来就 OVERWRITE=1,或者自己挪走): $1"; miss=1; fi
}
preflight() {
  local s d free
  echo "[preflight] 输入检查(只读)"
  chk_x "$PY"
  for s in pool_panels score_all_factors build_champion tradable_backtest blend_signals sweep_table \
           select_factors combine_two_level long_only_grid universe; do chk_file "scripts/$s.py"; done
  chk_dir "$STORE"; chk_file "$MKT"; chk_file "$CHAMPION"
  if want a; then
    for d in $POOL_SRCS; do chk_dir "$d/panels"; chk_file "$d/manifest.csv"; chk_file "$d/dates.npy"; done
  else
    chk_dir "$P/panels"; chk_file "$P/manifest.csv"; chk_file "$P/dates.npy"
  fi
  want c && ! want b && chk_file "$P/metrics.csv"
  want d && ! want c && chk_file "$SIG"
  want e && ! want c && chk_file "$SIG"
  want f && { chk_file "$REF_CHAMP"; chk_file "$REF_LEGACY"; chk_file "$REF_CHAMP_OLD"; }
  want f && ! want d && chk_file "$SW_POOL"
  want f && ! want e && chk_file "$SW_BLEND"
  want b && { chk_new "$P/metrics.csv"; chk_new "$P/fp_train_250x512.npy"; }
  want c && { chk_new "$SIG"; chk_new "$FAM.npy"; chk_new "$FAM.json"; }
  want d && chk_new "$SW_POOL"
  want e && { chk_new "$BLEND"; chk_new "$SW_BLEND"; }
  free=$(df -Pk "$PROJ" | awk 'NR==2{printf "%d", $4/1048576}')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then echo "  磁盘只剩 ${free} GB(< MIN_FREE_GB=$MIN_FREE_GB;families.npy 就要 4 GB)"; miss=1; fi
  if [ "$miss" = 0 ]; then echo "[preflight] OK(磁盘剩 ${free:-?} GB)"; else echo "[preflight] 有问题,不启动"; fi
  return $miss
}

# ---------------------------------------------------------------- 内存门槛
anon_gb() { awk '/^anon /{printf "%.1f", $2/1073741824}' "$MEMSTAT" 2>/dev/null; }
fits()    { awk -v a="$1" -v n="$2" -v l="$ANON_LIMIT_GB" 'BEGIN{exit !(a+n<=l)}'; }
gate() {      # gate <步骤名> <这一步预计要的匿名内存 GB>
  local name=$1 need=$2 a v waited=0
  [ "$MEM_GATE" = 1 ] || return 0
  a=$(anon_gb)
  if [ -z "$a" ]; then echo "[gate] 读不到 $MEMSTAT,$name 不设内存门槛"; return 0; fi
  if [ "$DRY_RUN" = 1 ]; then
    if fits "$a" "$need"; then v=放行; else v="会等(别的任务占着)"; fi
    echo "[dry-run] gate $name:容器匿名内存现在 ${a} GB + 这一步预计 ${need} GB,上限 ${ANON_LIMIT_GB} GB -> $v"
    return 0
  fi
  while ! fits "$a" "$need"; do
    if [ $(( waited / 60 )) -ge "$MEM_WAIT_MAX_MIN" ]; then
      echo "[gate] $name 等了 $(( waited / 60 )) 分钟,匿名内存还是 ${a} GB(要 ${need} GB,上限 ${ANON_LIMIT_GB} GB),放弃"
      exit 4
    fi
    if [ $(( waited % 600 )) -eq 0 ]; then
      echo "[$(ts)] [gate] $name 等内存:现在 ${a} GB + 预计 ${need} GB > 上限 ${ANON_LIMIT_GB} GB(已等 $(( waited / 60 )) 分钟)"
    fi
    sleep "$MEM_POLL_SEC"; waited=$(( waited + MEM_POLL_SEC )); a=$(anon_gb)
  done
  echo "[$(ts)] [gate] $name 放行:匿名内存 ${a:-?} GB + 预计 ${need} GB <= ${ANON_LIMIT_GB} GB"
}

# ---------------------------------------------------------------- 跑一步
shq() {       # 打印用:把一个参数引成能原样粘回 shell 的样子
  case $1 in
    '') printf "''" ;;
    *[!A-Za-z0-9_./=,:+@%-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
run() {       # run <步骤名> <日志文件 | - 表示输出留在主日志> <命令...>:每步前后各一行,退出码非 0 就整条停
  local name=$1 log=$2; shift 2
  local line="nice -n $NICE" a rc t0
  for a in "$@"; do line="$line $(shq "$a")"; done
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] step $name  (输出 -> $([ "$log" = - ] && echo 主日志 || echo "$log"))"
    echo "    $line"
    return 0
  fi
  echo "[$(ts)] step $name:$([ "$log" = - ] || echo " (输出 -> $log)") $line"
  t0=$SECONDS
  if [ "$log" = - ]; then nice -n "$NICE" "$@"; else nice -n "$NICE" "$@" > "$log" 2>&1; fi
  rc=$?
  echo "[$(ts)] step $name 用时 $(( SECONDS - t0 )) 秒"
  echo "[step $name exited with code $rc]"
  [ $rc -eq 0 ] || exit $rc
}
after() { [ "$DRY_RUN" = 1 ] || "$@"; }       # 跑完一步以后看一眼它的日志:预演时没有日志,跳过

echo "[$(ts)] h3_pipeline:TAG=$TAG  步骤 $FROM~$TO  DRY_RUN=$DRY_RUN  THREADS=$THREADS(打分 $SCORE_WORKERS 进程,回测 $SWEEP_WORKERS 进程)" \
     " NICE=$NICE  内存门槛 $([ "$MEM_GATE" = 1 ] && echo "开,上限 ${ANON_LIMIT_GB} GB" || echo 关)  OVERWRITE=$OVERWRITE"
echo "[$(ts)] 池子来源(按顺序,去重先到先得):$POOL_SRCS"
preflight || exit 3
echo "环境:export $ENV_LINE"

if [ "$DRY_RUN" != 1 ]; then
  mkdir "$LOCK" 2>/dev/null || { echo "[lock] $LOCK 已存在:另一个 h3 流水线在跑,或者上一次被 kill -9 留下的;确认后 rmdir $LOCK 再来"; exit 5; }
  LOCKED=1
  mkdir -p "$P" "$T"
fi

SRC_ARGS=""
for d in $POOL_SRCS; do SRC_ARGS="$SRC_ARGS --src $d"; done

# a 并池:只写 manifest 和软链;输出目录里已经有一个不一样的 manifest 会拒绝(来源变了就走这条),一样的重跑没事
if want a; then
  run pool - "$PY" scripts/pool_panels.py $SRC_ARGS --out "$P"
fi

# b 打分:和 _bench/trainto/pipeline.sh 里 legacy_full / mined_full 的打分同一组参数(全市场、剔除封板的标签、不写 rows.npy)
if want b; then
  gate score $(( SCORE_WORKERS * 2 + 3 ))
  if [ "$OVERWRITE" = 1 ] && [ -e "$P/fp_train_250x512.npy" ]; then
    # 训练段指纹是按 metrics.csv 的健康因子清单建的缓存,重新打分以后可能对不上:挪开(不删),build_champion 会重建
    if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] mv $P/fp_train_250x512.npy $P/fp_train_250x512.stale_<时间戳>.npy"
    else mv "$P/fp_train_250x512.npy" "$P/fp_train_250x512.stale_$(date +%s).npy"; fi
  fi
  run score "$P/score.log" "$PY" -u scripts/score_all_factors.py "$P" --buyable --no-rows --workers "$SCORE_WORKERS"
  after tail -n 1 "$P/score.log"
fi

# c 合成:和 legacy_signal.npy 同一组参数(--buyable,全市场);对照只放现冠军(路径里有 buyable,用的就是剔除封板的标签)
if want c; then
  gate build 20
  run build "$P/build.log" "$PY" -u scripts/build_champion.py --dir "$P" --buyable --out "$SIG" \
      --compare "冠军=$CHAMPION"
  after tail -n 12 "$P/build.log"
fi

# d 池子信号的 36 格回测(回测的并行在进程上,numpy 线程压到 1,免得 12 x 12)
if want d; then
  gate sweep_pool 6
  run sweep_pool "${SW_POOL%.json}.log" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" -u scripts/tradable_backtest.py sweep --inputs "$MKT" --signal "$SIG" --cost retail $G36 \
      --workers "$SWEEP_WORKERS" --out "$SW_POOL"
  after grep "^第 1 轮训练段最优" "${SW_POOL%.json}.log"
fi

# e 和现冠军等权混合(逐日秩 z 分数平均,不拟合权重),再扫同一套 36 格
if want e; then
  gate blend 3
  run blend - "$PY" scripts/blend_signals.py --signal "$SIG" --signal "$CHAMPION" --out "$BLEND"
  gate sweep_blend 6
  run sweep_blend "${SW_BLEND%.json}.log" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" -u scripts/tradable_backtest.py sweep --inputs "$MKT" --signal "$BLEND" --cost retail $G36 \
      --workers "$SWEEP_WORKERS" --out "$SW_BLEND"
  after grep "^第 1 轮训练段最优" "${SW_BLEND%.json}.log"
fi

# f 汇总:四个信号一张表(外加修复前的旧冠军,+2.28 出自它)
if want f; then
  echo "表里的名字:full36_${TAG}_signal = 新池子的合成信号;full36_blend_${TAG}_champion = 它和现冠军等权混合;"
  echo "  mined_full_G36_new = 现冠军(_bench/trainto,换手修复后);legacy_full_G36_new = 手写库(同上);mined_full_G36_old = 修复前的冠军(+2.28)"
  run table - "$PY" scripts/sweep_table.py "$SW_POOL" "$SW_BLEND" "$REF_CHAMP" "$REF_LEGACY" "$REF_CHAMP_OLD"
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] 预演结束:上面的命令一个都没有执行"
else
  echo "[$(ts)] 做完了($FROM~$TO)。信号 $SIG,混合 $BLEND,回测 $SW_POOL / $SW_BLEND"
fi
