#!/bin/bash
# One status line every 5 minutes into <run-id>_report.log (on the server).
# Usage: scripts/gp_report_loop.sh <run-id>
# anon = the container's real memory (cgroup memory.stat); ps RSS also counts the
# reclaimable mmap'd store pages and reads ~70 GB when anon is ~20 GB. Limit: 90 GiB.
# md0 = the data-disk RAID state; "write-pending" for minutes means writes are hung.
RUN_ID="$1"
[ -n "$RUN_ID" ] || { echo "usage: $0 <run-id>"; exit 2; }
cd /root/autodl-tmp/AlphaCube || exit 1
while true; do
  PID=$(ps -eo pid,args | grep -v grep | grep "run-id $RUN_ID" | awk '{print $1}' | head -1)
  if [ -n "$PID" ]; then
    ST="running up=$(ps -o etime= -p $PID | tr -d ' ') rss=$(ps -o rss= -p $PID | awk '{printf "%.1fGB",$1/1048576}') state=$(awk '/^State/{print $2}' /proc/$PID/status)"
  else
    ST="*** NOT RUNNING ***"
  fi
  ANON=$(awk '/^anon /{printf "%.1fGB",$2/2^30}' /sys/fs/cgroup/memory.stat)
  echo "[$(date '+%m-%d %H:%M:%S')] $ST anon=$ANON/90GB | gens=$(grep -cE '^ +[0-9]+/[0-9]+' ${RUN_ID}_run.log 2>/dev/null) | ckpt=$(ls mining_runs/$RUN_ID/checkpoints/*.csv 2>/dev/null | wc -l) | gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ') | sysdisk=$(df -h / | awk 'NR==2{print $5}') | md0=$(cat /sys/block/md0/md/array_state)" >> ${RUN_ID}_report.log
  sleep 300
done
