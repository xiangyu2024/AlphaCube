#!/bin/bash
# 连跑:一轮正常退出就自动并入它新封的结构和存档,立刻起下一轮(服务器上用)。
#   scripts/gp_chain.sh <下一轮编号> [最多轮数=6] [先等这个 run 跑完]
# 例:  screen -dmS chain bash -c "./scripts/gp_chain.sh 17 6 gpu16"
#
# 为什么要有它:每轮之间的动作是固定的(并封禁键、并存档、起下一轮),但要人来做
# 就有空转 —— gpu15 20:36 退出、gpu16 00:17 才起,白等三个半小时。
# 出任何问题都停下来并写 _bench/chain_halt,等人修:非 0 退出码、没有最终提交
# (members.csv)、archive 少于 100 条,都算问题。正常每轮一行写进 _bench/chain.log。
set -u
cd /root/autodl-tmp/AlphaCube || exit 1
N="${1:?用法: gp_chain.sh <下一轮编号> [最多轮数] [先等谁]}"
MAX="${2:-6}"
ADOPT="${3:-}"
PY=/root/miniconda3/envs/quant/bin/python

note() { echo "[chain $(date +%m-%d_%H:%M:%S)] $*" >> _bench/chain.log; }
halt() { note "停下: $*"; echo "$(date +%m-%d_%H:%M:%S) $*" > _bench/chain_halt; exit 1; }

fold() {                      # 一轮的收尾检查 + 并入下一轮
  local rid="$1" arc n
  grep -q "\[$rid exit 0 " "${rid}_run.log" 2>/dev/null \
    || halt "$rid 退出码非 0:$(grep -oE "\[$rid exit [0-9]+" "${rid}_run.log" 2>/dev/null | tail -1)"
  [ -f "mining_runs/$rid/members.csv" ] || halt "$rid 没写出最终提交(members.csv)"
  arc=$(ls mining_runs/"$rid"/checkpoints/archive_gen*.csv 2>/dev/null | sort | tail -1)
  [ -n "$arc" ] || halt "$rid 没有 checkpoint"
  n=$(( $(wc -l < "$arc") - 1 ))
  [ "$n" -ge 100 ] || halt "$rid 的 archive 只有 $n 条,不正常"
  PYTHONPATH=src "$PY" scripts/chain_next.py "${rid}_run.log" scripts/gp_full_market.sh "$arc" >> _bench/chain.log 2>&1 \
    || halt "$rid 并入下一轮失败"
  note "$rid 正常结束,archive $n 条,已并入下一轮"
}

rm -f _bench/chain_halt
if [ -n "$ADOPT" ]; then
  note "等 $ADOPT 跑完再接管"
  while pgrep -f "run-id $ADOPT" > /dev/null; do sleep 60; done
  sleep 10                      # 让它把最后一行日志和 members.csv 落盘
  fold "$ADOPT"
fi

for _ in $(seq 1 "$MAX"); do
  RID="gpu$N"
  [ -e "mining_runs/$RID" ] && halt "mining_runs/$RID 已存在,换个编号"
  note "启动 $RID"
  screen -S "${RID}rep" -X quit 2>/dev/null
  screen -dmS "${RID}rep" bash -c "./scripts/gp_report_loop.sh $RID"
  ./scripts/gp_full_market.sh "$RID" > "${RID}_run.log" 2>&1
  fold "$RID"
  N=$((N + 1))
done
note "连跑结束(已跑满 $MAX 轮)"
