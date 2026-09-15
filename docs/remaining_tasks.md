# 剩余任务清单(2026-09-14 晚)—— 完成状态见文末

状态:GP、四个 RL miner、池、奖励目标、研究入口、legacy 语料、种子库、下游五个阶段已跑在三维底层上,
273 个测试全绿。下面是尚未完成的部分,按"能否本机验证"分两组。

## A. 本机可完成并验证

| # | 任务 | 内容 | 验收 |
|---|---|---|---|
| 1 | 增量分段 store | 服务器生产 store 是"历史基础 store + 每月增量分段 + index.json"的组合(前项目 `segmented_store.py`),现在 `CubeStore` 只读单一目录。移植 `SegmentedCubeStore`,按日区间从多个来源拼接切片,契约哈希校验 | 合成一个基础 + 两个分段的 root,拼接读取与整块 store 逐 bit 相同;引擎结果逐 bit 相同 |
| 2 | D 主序磁盘副本 | NT 面按日分块读 S 主序文件是跨股票条带读。加 `(D, S, M)` 镜像文件的构建器,`CubeView.read_d` 有镜像时自动走镜像 | 有/无镜像读取逐 bit 相同;引擎结果逐 bit 相同 |
| 3 | 掩码策略 | 现在每个计算节点出口都施加停牌掩码。前项目默认只在叶子、`cross_*` 输入和根处掩(日内节点只影响同一股票日,假值会在越日或落地时被重新掩掉)。加 `mask_policy="all" / "boundaries"`,两者结果必须一致,默认取快的那个 | 随机文法树上两种策略逐 bit 相同 |
| 4 | 生产计算入口 + 断点续算 | 前项目 `run_factor_calculation`:按批算因子、每个因子一个原子写入的 `.npy`、目录级 catalog(OK / PARSE_FAILED / EVAL_FAILED / NOT_DAILY / DEFERRED)、资源风险算子延后策略、重启后跳过已完成因子。加 parquet 导出(日频流水线消费的格式) | 只跑一批后中断再续跑,产物齐全且逐 bit 相同;parquet 往返一致 |
| 5 | Feature Boosting 预压榨 | 前项目从用户设计稿实现的 O(n) 字段组 × 算子组随机配对 + GBDT 置换重要性筛选,幸存者进池。尚未移植 | 植入信号的合成立方体上,幸存者里出现 `day_std(Ret)` 类因子;进池后池 IC 为正 |
| 6 | `train.py` 全参数入口 + miner 对比脚本 | 前项目 `train.py`(所有参数显式列在文件末尾)和 `scripts/compare_all.py`(同一数据上跑全部 miner 出排行榜) | `python train.py` 在合成 store 上跑通;对比脚本出表 |
| 7 | 两轴/三轴算子 | 判断:分钟级跨股票配对(S² 量级)已由 `nt_ts_pearson`/`nt_ts_spearman` 覆盖(逐日 (M, S) 面);"日内市场 beta"这类看似三轴的因子可由面切换组合(`in_regbeta(Ret, nt_cs_mean(Ret), w)`)。不新增核函数,用测试证明组合可达并写进文档 | 组合因子在一个 D 轮内求值成功,与逐日参考一致 |
| 8 | 派生日频字段 vs V2 日频 store 比对脚本 | 前项目日频 store 的 Open/High/Low/Close/Volume/Amount/Vwap/Return 口径写在 `qmt_ingest`;写比对脚本,本机用同一立方体按该口径造一份日频 store 自校验,服务器上跑真数据 | 本机自校验逐 bit 相同;脚本可在服务器直接运行 |
| 9 | `check_store.py` | 打开 store、打印轴、字段、可交易占比、抽样读 | 合成 store 上运行 |

## B. 需要服务器

| # | 任务 | 内容 |
|---|---|---|
| 10 | CUDA 验证 | 写 `scripts/validate_gpu_parity.py`:在合成立方体上对每个注册算子跑 CPU 与 CUDA,报最大 ULP 差;整条 `mine`/`research` 加 `--device cuda` 跑通。本机无 CUDA,只能保证脚本可运行 |
| 11 | 真数据比对 | 任务 8 的脚本在 `/root/autodl-tmp/qmt_daily_data/fast_factor_compute` 上跑,确认派生日频字段与生产日频 store 一致 |

执行顺序:1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 脚本。每项带测试,做完更新 design.md 与 README。


## 完成状态(2026-09-14 深夜)

| # | 状态 | 落点 |
|---|---|---|
| 1 增量分段 store | 完成 | `store.SegmentedCubeStore` / `write_segment` / `open_cube_store`;tests/test_store_layouts.py |
| 2 D 主序镜像 | 完成 | `store.build_d_major_mirror`,`read_cube` 自动选路;同上 |
| 3 掩码策略 | 完成 | `CubeEngine(mask_policy="boundaries")` 默认,与 "all" 逐 bit 相同;tests/test_mask_policy.py |
| 4 生产入口 + 断点续算 + Parquet | 完成 | `calculate.py`,`cli calculate`;tests/test_production.py |
| 5 Feature Boosting | 完成 | `mining/boosting.py`,`run_mining(presqueeze=)`;tests/test_production.py |
| 6 train.py + miner 对比 | 完成 | `train.py`、`pipeline.run_mining`、`scripts/compare_miners.py` |
| 7 两轴/三轴算子 | 以组合覆盖 | tests/test_production.py::test_two_axis_factor_by_composition,design.md §10 |
| 8 日频比对脚本 | 完成(本机自校验) | `scripts/compare_daily_store.py`(`--self-check` OK;`--root` 待服务器) |
| 9 check_store | 完成 | `scripts/check_store.py` |
| 10 CUDA 验证 | 本机已用 MPS 实跑并修掉发现的问题;CUDA 待服务器复跑 | `scripts/validate_gpu_parity.py`、`tests/test_device_parity.py`、`CubeEngine(dtype="float64")` |
| 11 真数据比对 | `--root` 分支已在造出来的 V2 日频 store 上跑通;真数据待服务器 | `scripts/compare_daily_store.py`、`tests/test_compare_daily_store.py` |


## 第四轮(2026-09-14 更晚):把 10、11 从"只有脚本"推到"本机能验的都验了"

本机没有 CUDA,但有 MPS。任务 10 的实质是"引擎的非 CPU 路径有没有设备类缺陷",这一点换个
加速器一样能问,于是把脚本从写死 CUDA 改成 `--device`,在 MPS 上真跑了一遍全部 **238 个可搜索
算子**。结果:**0 个只在加速器上失败、0 个后端缺核、0 处有限/NaN 版图变化**——引擎本身没有
设备错配类缺陷(漏 `device=`、`.numpy()`、主机端索引)。

过程中三件事值得记下来,因为它们都是"看起来像结论、其实是量错了"的例子:

1. **ULP 要记在值来源的量级上**。`in_demean` 的绝对差 7.6e-06 一度被算成"相对误差 133%",
   因为它的输出被 `day_mean` 包成了解析意义上的 0。改用 `day_last` 收口(选择而非归约),
   并把 ULP 参照量级下限设在面板 99 分位,7.6e-06 就还原成它本来的样子:100 量级上的 1 个 ULP。
2. **光看 ULP 数判不了对错**。若干核函数经灾难性抵消得结果,自身 float32 误差就有上万 ULP
   (`nt_ts_hurst` 11296 ULP ≈ 相对 1.3e-3)。所以给引擎加了 `dtype="float64"` 参照模式,
   表格改成报**两个设备各自离 float64 多远**,判据是加速器误差是否超 CPU 的 `--slack` 倍。
3. **探针的操作数必须两个设备完全相同**。原先 ND 面探针喂 `day_mean(Close)`,两设备输入就已不同,
   再被 `ts_regbeta` 这类病态核函数放大 70 倍,看着像设备故障;更糟的是 `day_mean` 内部
   `x.float()` 收窄,float64 参照恰好继承 CPU 的舍入,比较对 GPU 不公平。换成 `day_last` 后,
   8 个"可疑"算子里 7 个消失。

剩下的第 8 个是真的:**`common.rank` 走非稳定 `argsort`,并列值的名次在不同后端互换**(1/n 级,
不是 ULP 级)。经确认后给七处排序加了 `stable=True`(见 design.md §5),这是对 vendored 核函数库
唯一一处自觉修改;改完 238 个算子全部落在归约顺序噪声内,legacy 语料验收数字不变。

任务 11 这边,`--root` 分支此前从未执行过。现在造了一份 V2 日频 store 做测试夹具,并且**两个轴
都造得比视图宽**(前后补日期、交错插入多余股票),逼出按标签而不是按位置对齐;轴对不上、字段缺失、
字段被污染各有一条用例。脚本也从"只打印"改成返回 0/1,服务器上可直接当判据用。
