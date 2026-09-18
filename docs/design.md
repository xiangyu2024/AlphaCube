# AlphaCube 设计:一个三维张量,三张面,一个引擎,全流程跑在它上面

## 1. 起点

数据天然是三维的:`(S, D, M)` = (股票, 交易日, 分钟)。前一个项目把它拆成两个二维世界:
DT 世界在固定股票的 `(D, M)` 面上算,ND 世界在分钟轴已归约的 `(D, S)` 面上算,
两套字段注册表同名不同义、两个解析器、两条引擎,以及为了把 DT 结果搬进 ND 世界的
`__dtN` 物化胶水。冗余在类型层,不在计算层;真正丢掉的是表达力:没有任何算子用到
第三张面 `(M, S)`,"每分钟跨股票排序再日内聚合"这类日内截面因子写不出来。

AlphaCube 只保留一个数据对象(立方体)、一份注册表、一个解析器、一个引擎,并把原项目的
全部功能迁到它上面:GP、四个 RL miner、PPO 热启动、三种池、奖励目标、研究入口、legacy
表达式(GeneralCall)、1498 条历史因子、种子库、组合层、模型层、组合构建/优化/执行、回测。

## 2. 三张面与算子的双份注册

同一个二维核函数喂 `(股票 × 天)` 和喂 `(股票 × 分钟)` 语义不同。在三维上要无歧义,
就必须把 ND 核函数**复制成两份注册项**,各自声明作用面:

| 面 | 核函数看到的 (行, 列) | 计算轴 | 执行模式 | 注册名 |
|---|---|---|---|---|
| dt / in | `(D, M)`,固定 s | M | 按股票或按日分块 | `in_*` |
| dt / day | `(D, M) → (D, 1)` | M(归约) | 按股票或按日分块 | `day_*` |
| dt / cross | `(D, M)`,固定 s | D | 按股票分块,逐股票调核 | `cross_*` |
| nd / cs | `(D, S)`,M 已归约 | S | 整面板 | `cs_*` |
| nd / ts | `(D, S)`,M 已归约 | D | 整面板 | `ts_*` |
| nt / cs | `(M, S)`,固定 d | S | 按日分块,**逐日调核** | `nt_cs_*` |
| nt / ts | `(M, S)`,固定 d | M | 按日分块,**逐日调核** | `nt_ts_*` |
| shared | elementwise | 无 | 任意 | `add` … |

`catalog.build_registry` 从 vendored 的核函数库(`src/alphacube/operators/`,与前项目逐字节
相同,**唯一一处自觉偏离见 §5 的稳定排序**)生成搜索注册表:DT 121 个、ND 48 个、NT 48 个(与 ND 同一函数对象)、共享 25 个,
共 242 个。`rank`(与 cs_rank 重复)与 `indneutralize`(需行业码)刻意排除。参数烘焙变体
沿用前项目。布尔输出(非根)与前视算子(不进搜索词表)由探针在构建期判定。

NT 面**逐日调核**是定义本身:把 ND 核函数作用在每一天的 `(M, S)` 面上。把多天的列拼成
一次调用更快,但不是同一个计算:两个 ND 核(`ts_pearson`/`ts_spearman`)本来就跨列混合,
而且任何归约的 float32 顺序都随张量几何变化,只有逐日调用既语义准确又与分块无关。

## 3. 静态类型:宽度与信息分开追踪

每个节点三个属性(`expression.Shape`):

- `collapsed`:分钟宽度字面为 1(`day_*` 输出、ND 面板、宽度 1 的 `in_slice`)。
- `degenerate`:逐分钟无真实变化,无论宽度。这是"每股票每天一个值"的真正含义。
- `width`:字面分钟宽度(`in_slice(x, 10, 229)` 是 220),混宽相遇在运行期报错。

规则:`in_*`/`nt_*`/`day_*` 要求操作数非退化;`in_expand` 要求字面宽度 1;ND 面算子要求
操作数退化(引擎给它 `(D, S)` 面板);`cross_*` 与 elementwise 透传/广播。**根必须退化**,
且不是布尔/近常数算子,且至少读一个字段。搜索文法严格执行;研究解析(`strict=False`)有
两处放宽:`day_*` 套在字面宽度 1 上(legacy 恒等/置零)、`in_expand` 套在任何退化操作数上
(引擎取第 0 列)。

日频字段不是叶子而是表达式:`close_1d = day_last(Close)`,`vol_1d = day_sum(Volume)`,
`ret_1d = close_1d / ts_delay(close_1d, 1) - 1`(`parser.DAILY_MACROS`)。`daily.DailyData`
(前项目 `FeatureData` 的替身)就是用这些表达式从立方体算出来的日频面板集合,下游阶段
消费它。

## 4. 引擎:按分块轴分阶段

节点的执行模式由计算轴决定(`engine/schedule.py`):`P`(整 `(D, S)` 面板驻内存:ND 面算子,
以及操作数全为面板/退化的 elementwise 节点)、`S`(按股票分块 `(C, D, M)`:D 轴算子必须)、
`D`(按日分块 `(S, C, M)`:NT 面必须)。自由节点先继承子节点模式,再继承父节点,默认 `S`。
轮次由就绪集贪心生成:能跑 `P` 先跑,否则取闭包更大的立方体模式。

**交接按轮次而不是按模式判定**:任何一条边的两端落在不同轮次(包括被另一模式隔开的
同模式,`S → D → S`),子节点就要交接 —— 退化值(含所有 P 节点)交接一个 `(S, D)` 面板
(取第 0 列,对退化值精确),全宽值以规范 `(S, D, M)` 顺序落盘,再按消费方朝向读回。
对现有算子库,不含 NT 算子的树恰好一次 S→P 交接。

**掩码策略**:`mask_policy="boundaries"`(默认)只在假有限值可能泄漏的位置重新施加停牌掩码:
D 轴算子(`cross_*`)与 NT 算子的输入,以及离开轮次的值(根、面板交接、落盘边界);日内节点只影响
自己的股票日,所以这与 `"all"`(每个节点都掩)结果逐 bit 相同(tests/test_mask_policy.py 在随机
文法树上锁定),但掩码开销降到原来的一小部分。

**store 的三种读法**(`store.py`,同一接口 `read_cube(name, rows, d0, d1)` / `read_status`):单目录
`CubeStore`;带 **D 主序镜像**(`build_d_major_mirror` 写出 `(D, S, M)` 副本,按日分块的多股票读取变成
连续读,读取结果逐 bit 相同);**增量分段** `SegmentedCubeStore`(生产布局:`<root>/minute` 历史基础 +
`<root>/incremental/<段>/minute` 追加段 + `index.json`,日期与股票在读切片时拼接,meta 哈希与 index
契约校验,新股票只在其出现的段里有值;与前项目 `segmented_store` 同一 schema)。`open_cube_store`
按目录结构自动选择。

**每块只读一次,而且只读切片**。一块之内,每个操作数(叶子字段、上一轮交接来的面板、落盘边界)
只物化一次,按引用计数在它最后一次被使用后释放。§6 的内存模型本来就按"叶子缓存 ≤ 8"来定块,
但此前的实现每**用**一次叶子就重读一次:gpu4 第 2 代的种群一块要读 104 次,实际只有 8 个字段。
共享同一张量的前提是**没有核函数写入自己的输入**(CSE 早就依赖这一点),由探针测试对全部
244 个注册算子逐 bit 锁定(`test_kernels_never_write_into_operands`)。`SegmentedCubeStore` 的
读取按"连续区间 → 切片"拼接;此前三个轴(连分钟轴)都走 `np.ix_`,逐元素 gather,在生产块形状
(15 只 × 1942 天)上比切片慢 21–28 倍(58 ms 对 2.1–2.7 ms)。两处合计:服务器 GPU 上同一批
129 棵树从 17.3 s 降到 5.6 s,叶子读取从占 67% 降到 5%,输出逐 bit 相同。

核函数视图(`engine/kernels.py`):M 轴 DT 核吃 `(A·B, M)` 零拷贝视图;`cross_*` 逐股票循环;
NT 面逐日 `(M, S)`;ND 面 `(D, S)`;elementwise 三维广播。常数是 `(1, 1, 1)` 视图。
`GeneralCall` 走同一套分派:字面参数按原位置重新插入,tensor-list 参数按位置组装,
list 值中间结果(multireg 的 tvalue 列表)只允许被另一个算子消费。

**dtype 契约**:每个节点出口都是 float32(bool/int 结果也转),与旧 numpy 路径"每节点
float 往返"一致;需要布尔掩码的核函数(参数名 `mask`/`*_mask`)在调用时按参数名转回 bool。
**停牌掩码**在每个计算节点出口施加(`is_gt0(NaN)` 返回有限的 0.0,只掩叶子挡不住它)。

## 5. 精度契约(测试锁定,见 tests/test_engine.py)

- DT 三族(in/day/cross,含 in_expand 与 elementwise 混合):与逐股票调核**逐 bit 相同**。
- NT 两族:与逐日在 `(M, S)` 面上调核**逐 bit 相同**(含跨列混合的 `nt_ts_spearman`)。
- ND 面:与直接在 `(D, S)` 上调核**逐 bit 相同**。
- 任意分块大小、任意轮次结构(含 S→D→S 落盘)结果**逐 bit 不变**;150 棵随机文法树上锁定。
  **这一条只在 CPU 上成立。** CUDA 按张量形状选归约路径,块的行数一变,`day_sum`/`day_mean` 就有
  ULP 级差异(实测 10~15% 的格点,相对 ~3e-7);对"舍入噪声过阈值"的核函数(`day_durbin_watson`
  的 `E[x²]-E[x]² < 1e-8`)会翻 NaN。选择类(`day_max`)不受影响。见 HANDOFF §8。
- 跨设备(CPU/CUDA/MPS)不承诺逐 bit,但承诺三件事(tests/test_device_parity.py,有加速器才跑):
  **没有算子只在加速器上失败**(设备错配、`.numpy()`、漏 `device=` 都表现为此)、
  **有限/NaN 版图不变**、**选择类算子**(`day_last` / `day_first` / `day_max` / `in_expand`,
  取元素而非归约)**逐 bit 相同**。

**float64 参照模式**:`CubeEngine(view, dtype="float64")` 把叶子、落盘、面板整条路径加宽。
这不是生产模式(两倍内存,GPU 后端不收 float64),而是给 float32 答案一个可测量的真值。
必要性:若干核函数经由灾难性抵消得到结果,它们**自身**的 float32 误差远大于任何设备差异——
在 16×30×241 合成立方体上 `nt_ts_hurst` 离 float64 有 11296 ULP(相对 1.3e-3)、`nt_ts_skew` 7913、
`in_zscore_rolling` 2950、`in_regres` 2742。所以"GPU 差了 3000 ULP"本身不能读作故障,
必须问"CPU 差了多少"。注意 `day_mean` 内部 `x.float()` 会把 float64 收窄回 float32,
这类核函数的参照值只是误差下界。

**稳定排序(对 vendored 核函数库的唯一一处自觉修改)**:`torch.argsort` / `torch.sort` 默认
**非稳定**,并列值的先后由后端自己决定。凡是输出"名次"或"用排序下标去取另一个张量"的核函数,
并列一旦被换序,结果就变了——这是 **1/n 级**差异,不是 ULP 级。真实分钟数据并列极多
(价格在最小变动价位格点上、成交量重复、停牌分钟为零),合成立方体里 Close 有 13.8% 的行、
Volume 有 44.6% 的行至少一处并列;实测 MPS 与 CPU 给同一行的并列值互换名次。

因此七处加上 `stable=True`,并列一律按下标先后(与 pandas `method="first"` 同约定):
`common.rank`(→ `in_rank` / `cs_rank` / `nt_cs_rank`)、`nd_ts` 的 spearman 窗口排名、
`legacy_specialized_dt` 的排名/近邻/取序下标三处、`legacy_specialized_nd` 的日期排序、
`dt_in` 的 `in_sorted_cumsum_{forward,backward}`(用 a 的排序下标去 gather b)。
**代价**:CPU 上并列处取值随之改变(合成盘 `in_rank` 影响 0.21% 的格点),与前项目的历史因子值
不再逐 bit 相同;legacy 语料的验收数字未变(1442 解析 / 1429 求值 / 1415 有限)。
**收益**:同一因子在 CPU 与 GPU 上取值相同、重复计算取值相同。
tests/test_device_parity.py 钉住这两条。

## 6. 内存模型

单位:一个 float32 立方体 = `S × D × M × 4 B`。峰值 ≈ (叶子缓存 ≤ 8 + 活跃中间量 ≈ 4 +
滚动窗口工作集 w) × 块立方体;`CubeEngine(chunk_bytes=...)` 按此自动定块(默认 2 GB)。
(叶子缓存直到 2026-09-18 才真正实现,见 §4;在那之前定块预算里有它,代码里没有。)

| 规模 | 单立方体 | 整体不分块 (w=60) | 64 只一块 |
|---|---|---|---|
| 300 × 970 × 241 | 0.28 GB | 20 GB | 3.7 GB |
| 1000 × 970 × 241 | 0.94 GB | 68 GB | 3.7 GB |
| 5215 × 2814 × 241 | 14.1 GB | 1.0 TB | 12.5 GB |

本机实测(300 × 500 × 241 合成,8 线程 CPU):8 条三面混合表达式 5.5 秒;GP 首代 24 棵
深度 4 的树 74 秒,后续每代 8 到 14 秒;平均每棵新树 1.1 秒。

## 7. 挖掘层:五个 miner 一套底层

**词表与文法**(`mining/tokens.py`):RPN token 空间 = 字段 + 可搜索算子(剔前视)+ 常数 +
窗口(各面窗口池的并集 + 面板真实分钟数)。窗口先于算子入栈,算子的掩码检查栈顶窗口是否
在自己的窗口池里(`in_expand` 只在真实分钟数下合法);操作数兼容性即 `operand_ok`;
`SEP` 只在单栈项是合法根时给出。掩码与 `push` 逐 token 一致(测试对全词表逐一比对)。

**延迟批量奖励**(`mining/env.py`):`TokenEnv.step` 在 SEP 时返回 `reward=None` 的待定
结果;miner 攒够一批(PPO 的 `episodes_per_update`、QCM 的 `episodes_per_batch`、SAGE 的
`trajectories_per_update`、Forge 的 decode 批)后 `BatchRewarder.rewards` 一次引擎调用求出
全部面板(批内 CSE),再按顺序过池:单因子 IC 有限 → 覆盖率 ≥ 阈值 → 与成员 |corr| 去重
→ 事务性插入 → 奖励 = 池 IC;任何拒绝 = 惩罚。`gamma = 1`、单终止奖励,回填后的 PPO 数值
与"立刻算"完全一致。这是 RL 走分钟数据唯一可行的形态。

**miner**:`AlphaGenMiner`(PPO,LSTM actor-critic)、`AlphaQCMMiner`(IQN,尾均值 CVaR 式选
动作,回放池按批推入,更新次数与逐步调度相同)、`AlphaForgeMiner`(Generator/Masker/
Predictor,直通估计)、`AlphaSAGEMiner`(RGCN 编码部分树 + GFlowNet 轨迹平衡,奖励 =
exp(clip(池 IC)) × 新颖度)、`PPOWarmStartQCMMiner`(PPO 记录带最终奖励的 transition,
QCM 回放池预热)、`GPMiner`(需求驱动的类型化生成器,面均衡抽算子,一代一次引擎调用,
覆盖率闸的 |IC| 适应度,结构聚类 + 簇剔除 + 簇内锦标赛,面板缓存跨代复用)。
各自与论文的简化处保留原项目的说明。

**池**(`mining/pool.py`):`MsePool`(z-score 后 OLS,缺失 z 视为中性 0)、`MeanStdPool`
(Adam 最大化 ICIR)、`DynamicRollingPool`(滚动窗口内拟合);同一套闸:IC 有限、覆盖率、
|corr| 去重、事务性;`test_ensemble` 用 label 的可执行收益作 P&L 腿。

**奖励目标**:`open_vwap15`(生产 label:t+1 日开盘 15 分钟 VWAP 进、t+1+h 出,直接从立方体
算;**价格空间见 §12**)、`close`,以及 `models.targets` 的 volatility / downside_volatility / rank / class /
spike / crash(`cli.make_label`)。

## 8. legacy 表达式:GeneralCall 与 1498 条历史因子

四类文法之外的调用走 `GeneralCall`(`fullcatalog.py` 扫描全库:装饰函数 + 各模块公开函数,
按模块与前缀归面;含 `legacy_framework_compat`、`legacy_specialized_*`、`dt_spectral`):
三张量(`select_where`)、字面参数(`in_slice`、`day_pdf_tail_sum_m(x, f, 50, False, 'rank')`)、
tensor-list(`in_multireg([...], y, d, 'tvalue')` → `tensor_list_std`)、改宽度(`delete_open_close`)。
调用前按核函数签名检查参数个数。别名:`abs`/`sum`/`rank`/`_max`/`_min`/`_pow`/
`cross_zscore_rolling`/`in_stddev`(按参数个数)/`in_multireg`(路由到 legacy 签名);语法糖:
`+ - * / ** | &`、`== > < >= <=`、`.float()`、list 参数、`in_expand(x)` 缺省宽度;因子按名
引用文本级内联;240 网格 → 241 网格迁移(`legacy/minute_grid.py`)。

**裸字段的日频语义**:表达式里没有任何分钟级算子(只有 ND 面与 elementwise)时,裸字段
一律取日频聚合(`ts_corr(open, vol, 10)` 是日频 open 与日频 vol),与旧 ND 模式解析一致;
含分钟算子的表达式里裸字段是分钟字段,ND 面算子的原始字段子树操作数才被强制日频化。

验收(`scripts/legacy_acceptance.py`,合成 6 × 60 × 241 立方体):

| 项目 | 数量 |
|---|---|
| 语料 | 1498 |
| 解析通过 | 1442 |
| 解析失败:字段不存在(行业、基本面、`*_1d` 分钟列) | 33 |
| 解析失败:库里没有的函数 `merge_and_forward_fill` | 8 |
| 解析失败:语料本身括号错位导致的参数个数错误 | 11 |
| 解析失败:分钟算子套在坍缩操作数上 | 4 |
| 根不是每股票每天一个值 | 13 |
| 求值通过 | 1429 / 1429 |
| 至少有有限值 | 1415 |

种子库 `configs/seed_alpha_library_{nd,dt}.yaml`:DT 51 条、ND 35 条全部可解析,
`legacy.seed_pool` 一次引擎调用后按序入池。

**生产计算入口**(`calculate.py`,`cli calculate`):按批求值、每个因子一个原子写入的 `.npy`、
目录级状态目录(SUCCESS / PARSE_FAILED / NOT_DAILY / EVAL_FAILED / DEFERRED_RESOURCE_RISK,含原因)、
资源风险算子延后策略(`ts_pearson`、`nt_ts_pearson`、`nt_ts_spearman`、`in_multireg` 及按名引用
它们的因子,`--allow-resource-risk` 放开)、**断点续算**(重启后按表达式摘要跳过已完成因子,
续跑结果与一次跑完逐 bit 相同)、Parquet 导出(`<根>/<YYYYMMDD>/<因子>.parquet`,列 `date, symbol,
<因子>`;上市前无行、停牌为空;manifest 可续)。

**Feature Boosting 预压榨**(`mining/boosting.py`):每轮配一对(字段组, 算子组),在携带的操作数上
做有界的局部组合,已是日频的候选一次引擎调用求值、GBDT 置换重要性排序,幸存者携带进下一轮;
未归约的全宽候选抽样携带作为操作数(`day_std(in_mean_rolling(Ret, 5))` 这类组合由此跨轮生成)。
`run_mining(presqueeze=...)` 把幸存者先送进池再开始搜索。

## 9. 下游阶段

`models/`(估计器、目标、评估、训练器、conformal CVaR、horizon mixture)、`combine/`
(post_combine 的多方法样本外排行榜与方向对齐,FactorGCL 超图 GNN)、`portfolio/`
(信号→权重、A 股手数优化器、美股优化器、凸稀疏优化)、`execution/`(Almgren-Chriss 与
鲁棒版)、`backtest/`(逐日引擎、成本、冲击、指标)从前项目整体迁入,数值逻辑逐字节不变,
只改了数据入口:它们只消费 `(D, S)` 面板 / `DailyData`,不再有任何表达式求值。唯一的
API 变化:`FactorGCLCombiner.fit(panels, target, epochs, train_days=)` / `predict(panels)`,
`build_prior_incidence(panels)`。

## 10. 边界与未做

- **两轴/三轴算子**:不新增核函数。分钟级跨股票配对(S² 量级)已由 `nt_ts_pearson` /
  `nt_ts_spearman` 覆盖(逐日 `(M, S)` 面);"日内市场 beta"这类看似三轴的因子由面切换组合可达,
  `day_last(in_regbeta(Ret, nt_cs_mean(Ret), 10))` 在一个 D 轮内求值且与逐日参考逐 bit 相同
  (tests/test_production.py::test_two_axis_factor_by_composition)。
- 语料里 11 条括号错位与 8 条 `merge_and_forward_fill` 无法计算(库里没有该函数)。
- **已知破绽:`sliding_{top,bottom}_{sum,mean}_masked_p5` 不满足分块不变性**。它们声明为 `dt/in`
  (分钟轴),但底层 `_sliding_windows` 沿 **dim 0** 滚动,而引擎在 dt/in 面喂进来的 `(A·B, M)` 里
  dim 0 是**股票日**——于是一只票的窗口会混进上一只票的行,结果随分块而变(首个差异恰好出现在
  第二个分块的第一只股票)。这不是 ULP 级差异。已用 strict xfail 钉住
  (tests/test_engine.py::test_sliding_masked_family_is_chunk_invariant),
  因为随机种群是否抽到这四个算子取决于词表大小,不能靠运气覆盖。修法待定:改核函数的滚动轴,
  或把它们改注册到 ND 面,或从注册表剔除。
- 引擎侧已修:归约前把**分钟轴的 stride-0 广播**落实(`engine/kernels._materialise`)。
  `in_first`/`in_last` 这类把一个值摊开的算子会产生 stride-0 视图,torch 在广播轴上归约时
  走哪条路径取决于张量整体大小,同一个股票日在 180 行与 600 行的块里会差一个 ULP。
- **加速器路径已在本机 MPS 上实跑**,不再是"只有脚本":`scripts/validate_gpu_parity.py --device mps`
  在 16×30×241 合成立方体上跑完全部 **238 个可搜索算子**,结果是
  **0 个只在加速器上失败、0 个后端缺核、0 处有限/NaN 版图变化**;值差异只有 `in_rank` 一处不属于
  归约顺序(见 §5 的并列排名)。这说明引擎的非 CPU 路径本身没有设备错配类缺陷。
  CUDA 仍需服务器复跑同一脚本(`--device cuda`,默认值),因为 CUDA 的归约实现与 MPS 不同。
- 脚本的读法:光看 ULP 数不能判断对错(见 §5 的 float64 参照),所以表格给的是
  **两个设备各自离 float64 参照多远**,而不是它们互相差多少;判据是加速器误差是否超过 CPU 误差的
  `--slack` 倍(默认 4)。探针的操作数与外包装一律用 `day_last`(选择而非归约):
  用 `day_mean` 会让两个设备拿到不同输入,再被病态核函数放大成看起来像设备故障的东西,
  而 float64 参照因 `day_mean` 内部收窄恰好继承 CPU 的舍入,比较就此对 GPU 不公平。
- `scripts/compare_daily_store.py --root <计算根目录>` 在真数据上比对派生日频字段与 V2 日频 store
  (本机自校验:Open/High/Low/Close 逐 bit,Volume/Amount/Vwap/Return 为 float32 求和顺序量级
  ~1e-7 的差异)。`--root` 分支本身已由 tests/test_compare_daily_store.py 覆盖:日频 store 的
  两个轴都造得比视图宽(前后补日期、交错插入多余股票),逼出**按标签而不是按位置对齐**;
  轴对不上、字段缺失、字段被污染各有一条用例,退出码 0/1 可直接作服务器上的判据。

## 11.4 平凡布尔算子不进有效动作集

根节点不得是布尔/近常数算子,但这条规则是**逐节点**检查的,于是搜索可以用任意算术把它洗白:
`pow_05(eq(day_ar1_coefficient(Vwap), day_cov(High, Vwap)))` 是合法的根,面板却几乎是常数指示量。
真实数据上它以 |IC| 0.58 占据种群第一,而它在池里记录的 IC 是 `nan`,并把 `equal`/`ic`/`icir`/
`lstsq` 四种组合方法的覆盖率全部拖成 0。

要挡的只是**会坍缩**的那一类。判据按算子的性质而不是一次采样的结果来定
(`catalog._probe_constant_capable`,喂全正、全负、混号、两操作数相同四组输入,任一组让输出变成
单一常数即判为平凡):

- **平凡(22 个,默认挡掉)**:对**固定阈值**取真假的谓词 —— 数据整体落在阈值一侧就全真或全假。
  `eq` / `greater` / `less` / `is_gt0` / `is_lt0` / `logical_{and,or,not,xor}` / `sign` /
  `*_sustain*` / `in_{is,no}_neighbor` / `in_left_one_distance_*` / `*_outlier_33`。
- **非平凡(7 个,保留)**:对**数据自身分布**取真假的谓词,无论输入如何都会把数据切开,
  因而携带真实截面信息 —— `cs_quantile_q{25,50,75}` / `nt_cs_quantile_q{25,50,75}` /
  `in_top_quantile_flag_q10`。

开关:`run_mining(allow_boolean_ops=...)` / `cli mine --allow-boolean-ops` / `train.py` 同名参数,
默认 `False`。GP 生成器、RL 词表、Feature Boosting 三处共用。真实数据实测(60 只,2023,GP):
挡掉后 best |IC| 从 0.58 回到 0.0422,池内成员 IC 全部有限,六种组合方法全部出数,
池 ICIR 从 +0.218 升到 +0.310。

## 11.45 配对核的代价:同一个核函数,两张面差两个数量级

`ts_pearson` / `ts_spearman` 的定义是"每只股票对其他每一只算滚动相关,再对 j 聚合",
时间复杂度 **O(窗口数 × d × N²)** —— 全市场 N=5219 时每个窗口要生成 5219×5219 的矩阵(2720 万对)。

**这是定义,不是实现绕远路。** 内存那一层曾经确实是代码问题(原实现建 `(T, d, N, N)` 字面张量,
全市场单日要 5.4 TB),已被重写为 `bmm` + 按时间分块,峰值内存变成固定预算。剩下的 O(N²) 躲不掉:
我验算过,协方差双线性,默认的 `agg="mean"` 在数学上可折叠成
`mean_j corr(i,j) = cov(x_i, Σ_j x_j/σ_j) / (σ_i·N)`,即 O(N);**但实现里 `num_valid` 是逐对统计
共同有效观测数的**(停牌与上市日期各不相同),该恒等式只在"窗口内所有股票都完整观测"时成立,真实
数据里不成立。

**真正决定负担得起与否的是面**:

| 注册项 | 面 | 求值方式 | 相对代价 |
|---|---|---|---|
| `ts_pearson` / `ts_spearman` | ND `(D, S)` | 整个面板一次 | 1× |
| `nt_ts_pearson` / `nt_ts_spearman` | NT `(M, S)` | **逐日 1942 次** | **约 241×** |

所以搜索词表只排除 NT 面那两个(外加 `in_multireg`),ND 面的保留 ——
一刀切会为了一张面的代价丢掉整个因子家族。`--allow-resource-risk` 可全放开。
`calculate.py` 延后的是全部四个,那是另一个(内存)问题,两者共用 `RESOURCE_RISK_KERNELS`
这一个定义,`SEARCH_FORBIDDEN_KERNELS` 是它的真子集。

实证:全市场首跑在第 14 代繁殖出一棵 NT 配对核的树,那一代跑了 **196 分钟**,而正常代是 4 分钟。

## 11.5 价格空间:复权价与原始成交额不能混用

2026-09-15 在真实 store 上首跑,GP 连续 20 代 `valid 0`。逐层定位(`scripts/diagnose_run.py`)
落在 label:`open_vwap15` 原先用 `Amt / Volume` 当价格,再用当日 `High`/`Low` 做区间校验,
**校验筛掉了 100% 的股票日**,label 全 NaN,于是每个候选的 IC 都是 NaN。

根因不是缺数据,是**价格空间**:生产 store 里 `Open/High/Low/Close/Vwap` 是**后复权价**,
而 `Amt`/`Volume` 是**原始成交额与股数**。判据是比值 `Close/(Amt/Vol)` 跨股票从 2.5 到 113
(差 45 倍),但**每只票内部变异系数只有 1%~2%** —— 单位换算会是一个全局常数,复权因子才会
每只票一个值。生产日频 store 自己也印证:它的 `Vwap/Close = 0.99994`、100% 落在 `[Low, High]`
内,而 `Amount/Volume` 是 0.108×Close。

**结论:`Amt` 只能用于成交额类因子,永远不要除以 `Volume` 去得到价格。** 两处按此改正:

- `label.open_vwap15`:改为对分钟 `Vwap` 字段按 `Volume` 加权。分子分母**共用一个掩码** ——
  有 5.5% 的分钟有成交量却没有 Vwap,只把它们的成交量计入分母会让结果系统性偏低(中位 5%)。
  同时不再要求窗口内 15 分钟全齐:生产数据上全齐只占 67.33%,而 ≥5 分钟占 99.98%
  且没有任何股票日少于 3 分钟,故 `OPEN_VWAP_MIN_MINUTES = 5`。改后价格腿有限占比 0% → 99.98%。
- `daily.DAILY_FIELD_EXPRESSIONS["Vwap"]` 与 `scripts/compare_daily_store.py` 的参照同样改正。
  表达式里用 `mul(0.0, ...)` 表达那个掩码:`0 * NaN = NaN`,缺 Vwap 的分钟因此从 NaN 安全的
  `day_sum` 里掉出去,其余分钟贡献 `Volume - 0`。

## 11. 与服务器数据的关系

`open_cube_store` 直接读前项目的 V2 计算根目录(`minute/` + `incremental/`)或单个分钟 store 目录,
无需转换;`python train.py`(参数全部显式)、`cli mine/research/calculate`、`scripts/check_store.py`、
`scripts/compare_miners.py` 都接受同一个 `--store`:

```bash
python -m alphacube.cli mine --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute/minute \
    --start 20230101 --end 20231231 --max-symbols 300 --miner alphaqcm --budget 2000 --device cuda \
    --seed-library configs/seed_alpha_library_dt.yaml --out mining_runs
```
