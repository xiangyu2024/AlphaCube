#!/bin/bash
# 原始手写因子库(1498 条)-> 求值 -> 打分 -> 相关性聚类 + 两层合成 -> 100 万纯多头真实回测,
# 全市场一份、沪深300 一份(服务器上用,每步退出码写进日志):
#   screen -dmS legacy bash -c "./scripts/legacy_pipeline.sh > _bench/legacy_pipeline.out 2>&1"
#
# 和挖掘因子那套(_bench/allfac_buyable、_bench/allfac_hs300)完全同一个口径:剔除封板的标签、训练段
# 指纹聚类(|秩相关| 0.7)、族内 ridge、族间 lstsq、半衰期训练段选;回测 A 股实际费率、整手、
# 上市不足 60 天不买,配置都在训练段选,测试段只报数。
set -u
cd /root/autodl-tmp/AlphaCube
export PYTHONPATH=src PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/quant/bin/python
U=_bench/universe/hs300_proxy_v1.npy
L=_bench/legacy
H=_bench/legacy_hs300
T=_bench/tradable
MKT=$T/market_20180102_20251231.npz
LOG=$L/pipeline.log
mkdir -p $L
step() { echo "[$(date '+%F %T')] $*" >> $LOG; }

step "1 求值(全市场生产视图)"
$PY -u scripts/eval_legacy_factors.py $L > $L/eval.log 2>&1
rc=$?; step "1 exit $rc"; [ $rc -eq 0 ] || exit $rc
# 和挖掘同卡时显存封顶,长窗口的 ts 运算(alpha_* 一次要 13-17 GB)会 OOM;失败的面板不落盘。
# 等同卡的挖掘结束,不封顶、一次一条把缺的补上
while pgrep -f "alphacube.cli mine" > /dev/null; do sleep 60; done
step "1b 补算(显卡空出来之后,不封顶)"
env -u ALPHACUBE_GPU_FRACTION $PY -u scripts/eval_legacy_factors.py $L --batch 1 > $L/eval2.log 2>&1
rc=$?; step "1b exit $rc;$(grep 完成 $L/eval2.log)"; [ $rc -eq 0 ] || exit $rc
# 一次一条、整张卡都装不下的(全市场 250 天窗口的 ts_corr),打分时按缺失处理(不健康,不进合成)

step "2 全市场打分"
$PY -u scripts/score_all_factors.py $L --buyable --no-rows --workers 8 > $L/score.log 2>&1
rc=$?; step "2 exit $rc"; [ $rc -eq 0 ] || exit $rc
step "3 全市场聚类合成"
$PY -u scripts/build_champion.py --dir $L --buyable --out _bench/legacy_signal.npy \
    --compare 挖掘冠军=_bench/champion_signal_buyable.npy > $L/build.log 2>&1
rc=$?; step "3 exit $rc"; [ $rc -eq 0 ] || exit $rc
step "4 全市场真实回测"
$PY -u scripts/tradable_backtest.py sweep --inputs $MKT --signal _bench/legacy_signal.npy --cost retail \
    --caps1 0.005,0.01 --halflives 5,10 --bands 0,0.5,1.0 --keeps 0,0.3,0.5 --stage1-only --workers 12 --top 15 \
    --out $T/sweep_legacy_full.json > $T/sweep_legacy_full.log 2>&1
step "4 exit $?"

mkdir -p $H
[ -e $H/manifest.csv ] || cp $L/manifest.csv $L/dates.npy $H/
[ -e $H/panels ] || ln -s /root/autodl-tmp/AlphaCube/$L/panels $H/panels
step "5 沪深300 打分"
$PY -u scripts/score_all_factors.py $H --buyable --no-rows --universe $U --workers 8 > $H/score.log 2>&1
rc=$?; step "5 exit $rc"; [ $rc -eq 0 ] || exit $rc
step "6 沪深300 聚类合成"
$PY -u scripts/build_champion.py --dir $H --buyable --universe $U --out _bench/legacy_signal_hs300.npy \
    --compare 挖掘因子沪深300版=_bench/champion_signal_hs300.npy > $H/build.log 2>&1
rc=$?; step "6 exit $rc"; [ $rc -eq 0 ] || exit $rc
step "7 沪深300 真实回测"
$PY -u scripts/tradable_backtest.py sweep --inputs $MKT --signal _bench/legacy_signal_hs300.npy --cost retail \
    --universe $U --pool 1.0 --caps1 0.01,0.02,0.03,0.05 --keeps 0,0.3,0.5 --bands 0,0.5 --halflives 5,10,20 \
    --stage1-only --workers 12 --top 15 --out $T/sweep_legacy_hs300.json > $T/sweep_legacy_hs300.log 2>&1
step "7 exit $?"

step "8 全样本净值(训练段选出的配置)"
for V in full hs300; do
  SIG=$([ $V = full ] && echo _bench/legacy_signal.npy || echo _bench/legacy_signal_hs300.npy)
  UARG=$([ $V = full ] && echo "" || echo "--universe $U --pool 1.0")
  NAME=$([ $V = full ] && echo 全市场 || echo 沪深300)
  ARGS=$($PY -c "import json; d=json.load(open('$T/sweep_legacy_$V.json')); k=d['best1_kw']; print(f\"--cap {k['cap']} --keep {k['keep']} --band {k['band']} --halflife {d['best1_hl']}\")")
  $PY -u scripts/tearsheet.py compute --signal $SIG $UARG $ARGS \
      --title "原始因子库,100 万纯多头($NAME):2018-2025 全样本(竖线右边是样本外)" \
      --out $T/fullsample_legacy_$V.json > $T/fullsample_legacy_$V.log 2>&1; step "8 $V exit $? ($ARGS)"
done
step "done"
