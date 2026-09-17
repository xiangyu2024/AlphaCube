# AlphaCube

把因子计算与挖掘从"两个二维世界(ND 日频面板 + DT 分钟面板)"改成**一个三维张量**
`(S, D, M)` = (股票, 交易日, 分钟),并把原项目的全部功能迁到这个底层上:GP、四个 RL miner
(AlphaGen / AlphaQCM / AlphaForge / AlphaSAGE)与 PPO 热启动、三种因子池、全部奖励目标、
研究入口、legacy 表达式(GeneralCall)与 1498 条历史因子、种子库、组合层、模型层、
组合构建 / 优化 / 执行、回测。设计与取舍见 [docs/design.md](docs/design.md)。

与前项目 AlphaFactorMining 的关系:核函数库 `src/alphacube/operators/` 原样 vendor;下游阶段
(models / combine / portfolio / execution / backtest)整体迁入、数值逻辑不变;其余全部新写。
store 目录格式与前项目的 V2 分钟 store 相同,服务器上的数据直接可读。

## 三张面

| 面 | 数据 | 算子 |
|---|---|---|
| DT | 固定股票的 (日, 分钟) | `in_*` 分钟内滚动 / `day_*` 归约分钟 / `cross_*` 跨日 |
| ND | 分钟已归约的 (日, 股票) | `cs_*` 截面 / `ts_*` 时序 |
| NT | 固定日的 (分钟, 股票) | `nt_cs_*` 每分钟跨股票截面 / `nt_ts_*` 分钟内时序 |

ND 与 NT 是同一批核函数的两份注册(同一个函数对象、不同的面):同一核函数喂"股票 × 天"
和"股票 × 分钟"语义不同,三维上必须由名字消歧。一个表达式里三张面可以混用:

```
day_mean(nt_cs_rank(Close))                       # 每分钟跨股票排名,再日内均值
cs_rank(ts_mean(day_std(Ret), 5))                 # 分钟波动 -> 日频时序均值 -> 日频截面排名
day_last(nt_cs_rank(cross_mean(Close, 3)))        # 跨日平滑后再做分钟级截面(一次面切换)
ret_1d                                            # 日频字段是表达式:close_1d / ts_delay(close_1d, 1) - 1
day_autocorr(in_slice(ret, 11, 230), 1)           # legacy 写法:字面参数、GeneralCall
```

## 快速开始

```bash
cd ~/Desktop/AlphaCube
# 依赖:numpy pandas torch scikit-learn scipy pyyaml pytest(本机 venv:~/.venvs/alphacube)
PYTHONPATH=src python -m pytest tests -q

# 合成一个立方体 store(--plant 植入一个已知信号)
PYTHONPATH=src python -m alphacube.cli synth --out /tmp/cube --symbols 60 --days 300 --plant 1.0

# 研究:表达式 -> 指标表 + 因子值(--legacy 加载 1498 条历史因子)
PYTHONPATH=src python -m alphacube.cli research --store /tmp/cube --start 20200301 \
    --expr "a=day_std(in_mean_rolling(Close, 30))" --expr "b=day_mean(nt_cs_rank(Ret))" --out /tmp/runs

# 挖掘:任意 miner、任意池、任意 label,可先用种子库预热池,最后出样本外组合排行榜
PYTHONPATH=src python -m alphacube.cli mine --store /tmp/cube --start 20200301 \
    --miner alphagen --pool meanstd --label open_vwap15 --budget 2000 \
    --seed-library configs/seed_alpha_library_dt.yaml --out /tmp/runs
PYTHONPATH=src python -m alphacube.cli mine --store /tmp/cube --miner gp --label volatility --horizon 5 --budget 480

# 生产计算(可断点续算)+ Parquet 导出
PYTHONPATH=src python -m alphacube.cli calculate --store /tmp/cube --legacy --out factor_runs --parquet /tmp/parquet

# 全参数入口(所有参数显式写在文件末尾)、体检、miner 对比、legacy 语料验收
ALPHACUBE_STORE=/tmp/cube python train.py
PYTHONPATH=src python scripts/check_store.py --store /tmp/cube
PYTHONPATH=src python scripts/compare_miners.py --store /tmp/cube --budget 200
PYTHONPATH=src python scripts/legacy_acceptance.py
```

`--store` 接受 V2 计算根目录(`minute/` + `incremental/` 增量分段)或单个分钟 store 目录。

`--miner` 取 gp / alphagen / alphaqcm / alphaforge / alphasage / warmstart;`--pool` 取
mse / meanstd / dynamic;`--label` 取 open_vwap15 / close / volatility / downside_volatility /
rank / class / spike / crash。服务器上把 `--store` 指向 V2 分钟 store,加 `--device cuda`。

## 目录

```
src/alphacube/
  axes.py / catalog.py        三根轴、三张面、OpSpec;搜索注册表(DT + ND + NT 双份 + 共享 + 变体)
  fullcatalog.py              全库可调用面(GeneralCall 用),别名与 legacy 路由
  expression.py / parser.py   表达式树与静态类型;字符串解析(严格 / legacy 两档)
  store.py / synthetic.py     (S, D, M) mmap store(单目录 / D 主序镜像 / 增量分段)与视图;合成立方体
  calculate.py / pipeline.py  生产计算(断点续算、状态目录、Parquet);一步挖掘流程(train.py / cli 共用)
  engine/                     dag(CSE) / schedule(模式与轮次) / kernels(各面视图) / engine(分块执行、掩码、物化)
  label.py / metrics.py       日频 label(open_vwap15 / close);IC、rank IC、ICIR、组合、回测指标
  daily.py                    DailyData:由立方体派生的日频面板(下游阶段的输入)
  research.py                 evaluate_factors:表达式 dict -> 指标表 + values/*.npy
  mining/                     tokens(RPN 文法) env(延迟批量奖励) alphagen alphaqcm alphaforge alphasage
                              warmstart gp generator clustering pool boosting(Feature Boosting 预压榨)
  legacy/                     1498 条历史因子、240→241 网格迁移、种子库
  models/ combine/ portfolio/ execution/ backtest/   从前项目迁入的下游阶段
  cli.py                      synth / research / mine
  operators/                  vendored 核函数库(仅一处自觉修改:排序加 stable,见 design.md §5)
tests/                        精度契约、分块不变性、调度、掩码、文法、RL、legacy 语料、下游各阶段
docs/design.md                设计文档
docs/HANDOFF.md               交接说明:当前进展、服务器怎么连/怎么跑/怎么监控、踩过的坑
```

## 验证

| 项目 | 结果 |
|---|---|
| 测试 | 303 passed, 8 skipped(可选依赖 lightgbm / xgboost / tabpfn) |
| 掩码策略 / 分段 store / D 主序镜像 / 断点续算 | 与整体掩码、整块 store、无镜像、一次跑完逐 bit 相同 |
| DT 三族 / NT 两族 / ND 面板 | 与逐股票 / 逐日 / 整面板直接调核逐 bit 相同 |
| 分块不变性 | 任意分块与轮次结构逐 bit 不变(150 棵随机文法树) |
| 植入信号 | 研究入口测得 IC 大于 0.25;GP 前 110 天挖出的池在后 50 天样本外 IC 大于 0.15 |
| legacy 语料 | 1442 / 1498 解析,1429 / 1429 求值,1415 条有有限值(其余为缺字段、语料括号错位、库中无此函数) |
| 加速器路径(本机 MPS 实跑) | 238 个可搜索算子:0 个只在加速器上失败、0 个后端缺核、0 处有限/NaN 版图变化、0 处超出归约顺序的值差异 |

## 未做

只剩需要服务器**复跑**的两项,两个脚本本机都已跑过(一个换设备,一个换数据):

```bash
PYTHONPATH=src python scripts/validate_gpu_parity.py --device cuda      # 本机已用 --device mps 跑通
PYTHONPATH=src python scripts/compare_daily_store.py --root /root/autodl-tmp/qmt_daily_data/fast_factor_compute
```

前者逐算子比对 CPU 与 CUDA,判据是各自离 float64 参照多远(见 design.md §5);后者比对派生日频字段
与 V2 日频 store,退出码 0/1 即结论。两轴/三轴算子由面切换组合覆盖,不新增核函数。详见 design.md §10。
