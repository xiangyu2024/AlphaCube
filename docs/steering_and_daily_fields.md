# 日频字段 + 搜索指引:怎么建库、怎么用

> 写给接手的新窗口。这一份讲 2026-10-06 这批改动**新增的能力**:表达式里能直接写
> 日频字段(市值 / 涨跌幅限制 / 板块 / 上市天数),以及外部插件能从外面**指引**搜索
> (算子权重、候选筛子、打分器、叶子槽位)。
>
> 设计文档在 `/Users/wxy/Desktop/Partura/docs/alphacube_steering_design.md`(1081 行,
> 含每条决定的实测依据与 11 个工作包);这一份只讲**怎么用**,每个例子都在本机跑过。
> 相关:`docs/HANDOFF.md`(引擎 / 挖掘的总交接)、`docs/可交易回测.md`(回测口径)。
>
> **一句话**:所有新东西的默认值都选中今天这条代码路径。不传新参数、不加 `--plugin`、
> 不给 `--daily-fields`,随机流和结果与改动前逐 bit 相同;可搜索算子数仍是 218 / 214,
> 算子总数仍是 244。

---

## 0. 五件新东西,一页看完

| | 是什么 | 入口 | 默认 |
|---|---|---|---|
| 日频字段 | 9 个只有日频、没有分钟定义的字段,可以当叶子写进表达式 | `--daily-fields DIR` / `daily_root=` | 不给目录时仍有 3 个派生字段 |
| 生成器关键字 | 算子权重、候选筛子、日频叶子、叶子槽位 | `TypedTreeGenerator(...)` / `SearchHooks` | 全部 `None` / `()`,不消耗随机数 |
| 打分器注册表 | 插件自己注册一个 `--fitness` 取值 | `mining.scoring.register_scorer` | 注册表为空,`make_scorer` 走老分支 |
| 插件加载 | 一个 importable 模块,带 `register()` / `search_hooks()` | `--plugin MODULE` / `$ALPHACUBE_PLUGINS` | 不加载任何插件 |
| `real_valued` | 声明某个核函数输出是实数,跳过布尔探针(探针会误判) | `@op(..., real_valued=True)` / `declare_real_valued()` | 谁都没声明,词表不动 |

---

## 1. 日频字段

### 1.1 九个字段,三类

| 类 | 字段 | 从哪来 | 要不要建库 |
|---|---|---|---|
| **stored** | `FloatCap` `TotalCap` `FloatShares` | `<daily_root>/daily_<name>.npy` + `meta.json`,磁盘上 float64 | **要**,见 §1.4 |
| **derived** | `LimitWidth` `BoardCode` `ListAge` | 从 store 自己的 symbols / dates / status 算出来 | **不要**,给个 store 就有 |
| **log** | `LogFloatCap` `LogTotalCap` `LogFloatShares` | stored 平面在 **float64 里**取 log,之后才降 float32 | 跟 stored 一起有 |

`log` 这一类不是方便,是**精度**:市值是 1e9~1e13 量级,float32 只有 ~7 位十进制,
`ts_delta(FloatCap, 1)` 是两个几乎相等的大数相减;`ts_delta(LogFloatCap, 1)` 就是对数收益,
表示噪声约 6e-8,对着 1e-3 的真实日内变动。**做差分、做比值,一律用 `Log*`。**

因果性:第 `d` 天的平面存的是**收盘时点可知的东西**。没有任何隐式 shift ——
想用昨天的市值就自己写 `ts_delay(FloatCap, 1)`。

### 1.2 语法:哪些算子吃日频叶子

日频叶子的 shape 复用的就是 `COLLAPSED` 这个常量 —— 也就是 `day_*` 输出已经带的那个
shape。所以"一天一个值"这件事,类型系统里本来就有的规则全都自动适用:

| 算子族 | 裸日频叶子 | 为什么 |
|---|---|---|
| `cs_*` / `ts_*`(**ND 面**) | **接受** | 引擎直接把 `(D, S)` 面板递给它 |
| `cross_*` | **接受** | 输出经 `_passthrough` 仍是 COLLAPSED |
| 逐元素(`shared`) | **接受** | 对着分钟张量会把当天这一个值**广播到全部 M 个分钟** |
| `day_*` / `in_*`,以及 `nt_*`(**NT 面**,要完整分钟轴) | **两种模式都拒** | 一天只有一个值,按分钟算没有意义 |
| `in_expand` | **接受**(strict 也接受) | 故意留的:它就是把当天的值铺到分钟轴,跟 `day_*` 的输出一样 |

注意最后两行的区别:**同一个核函数的 ND 面吃日频叶子,NT 面不吃**。
实测 `cs_rank(FloatCap)` 两种模式都 OK,而 `nt_cs_rank(FloatCap)` 两种模式都 `ParseError`
—— NT 面要的是完整分钟轴。(这和 §2 / §6.6 里"指引时两个名字都要写"是同一件事的两面。)

实测(`parse(strict=True)` / `parse_legacy` 都试过):

```
cs_rank(FloatCap)                       两种模式都 OK
ts_delta(LogFloatCap, 5)                两种模式都 OK
cross_mean(FloatCap, 5)                 两种模式都 OK
mul(Ret, LimitWidth)                    两种模式都 OK(FULL)
day_mean(mul(Ret, LimitWidth))          两种模式都 OK(COLLAPSED,可以当根)
FloatCap                                裸叶子也是合法的根
in_expand(FloatCap, 241)                OK,WIDE_DEGENERATE
day_sum(FloatCap)                       两种模式都 ParseError
in_mean_rolling(FloatCap, 5)            两种模式都 ParseError
nt_ts_mean(FloatCap, 5)                 两种模式都 ParseError
nt_cs_rank(FloatCap)                    两种模式都 ParseError(NT 面)
day_sum(log(FloatCap))                  legacy 接受(等于恒等),strict 拒 —— 见 §6.3
```

别名:`float_cap` / `floatcap` / `circ_mv` → `FloatCap`,`limit_width` → `LimitWidth`,
`log_float_cap` → `LogFloatCap`,等等,共 20 个(`parser.DAILY_FIELD_ALIASES`)。
**没有裸 `age`**,太容易撞。日频分支排在 `name()` 解析链的**最后**,所以将来谁加了一个
撞名的字段,不会悄悄盖掉已有的分钟字段。

### 1.3 最小可运行例子

**(a) 不建库,只用派生字段** —— 任何一个 store 都能直接跑:

```bash
cd /Users/wxy/Desktop/AlphaCube-steering
export PYTHONPATH=$PWD/src
~/.venvs/alphacube/bin/python -m alphacube.cli synth --out /tmp/demo/cube \
    --symbols 12 --days 40 --minutes 16
~/.venvs/alphacube/bin/python -m alphacube.cli research --store /tmp/demo/cube \
    --expr 'capw=cs_rank(mul(day_sum(Amt), LimitWidth))' \
    --label close --out /tmp/demo/runs --run-id d1
```

```python
from alphacube.store import open_cube_store
store = open_cube_store("/tmp/demo/cube")      # 没有 daily_root
view = store.view()
view.daily_fields          # ('LimitWidth', 'BoardCode', 'ListAge')
store.daily                # None
view.daily("LimitWidth")   # (S, D) float32,取值 [0.1 0.2 0.3]
```

**(b) 建一个小库,用 stored / log 字段**:

```python
from alphacube.synthetic import make_synthetic_cube, write_synthetic_store, write_daily_fields

write_synthetic_store("/tmp/demo/cube2", n_symbols=12, n_days=40, n_minutes=16)
_, symbols, dates, _, planes = make_synthetic_cube(
    n_symbols=12, n_days=40, n_minutes=16, daily_fields=("FloatCap", "TotalCap"))
write_daily_fields("/tmp/demo/daily_fields", planes, symbols, dates)
# 写出 daily_FloatCap.npy / daily_TotalCap.npy / meta.json
```

```bash
~/.venvs/alphacube/bin/python -m alphacube.cli research --store /tmp/demo/cube2 \
    --daily-fields /tmp/demo/daily_fields \
    --expr 'turnover=cs_rank(div(day_sum(Amt), FloatCap))' \
    --label close --out /tmp/demo/runs --run-id d2
```

跑出来的那两行(本机实测):

```
20200102..20200226: 40 days (+0 warm-up) x 12 symbols; 1 expressions;
  rounds: S[1] -> P[2]; boundaries: 0; panel hand-offs: 1
          ic_mean     icir  rank_ic coverage ls_sharpe
turnover  -0.0309  -0.1121  -0.0318   0.9812   -1.2735
```

注意 `div(day_sum(Amt), FloatCap)` 这个换手率因子的 `div` 节点**落在面板轮**
(`S[1] -> P[...]`),不是分钟轮 —— 设计阶段有方案把这件事说反过,这里是实测结论。
所以引擎里 `_run_panel_round` 的那个 `operand()` 分支是**必需**的,不是可选优化
(漏掉它,这一族因子就根本算不出来)。

`--daily-fields` / `--plugin` 在 `research` / `calculate` / `mine` 上都能用(公共父解析器),
库调用对应 `run_mining(..., daily_root=...)`、`evaluate_factors(..., daily_root=...)`、
`run_factor_calculation(..., daily_root=...)`。环境变量 `$ALPHACUBE_DAILY_FIELDS` 是
`open_cube_store` 的默认值,方便 sweep 脚本。

### 1.4 建库(服务器上跑一次)

```bash
python scripts/build_daily_fields.py \
    --store /root/autodl-tmp/qmt_daily_data/fast_factor_compute \
    --cap-dir /root/autodl-tmp/qmt_daily_data/daily_capital \
    --out /root/autodl-tmp/AlphaCube/daily_fields \
    --min-coverage 0.5 --dry-run        # 先 dry-run 看清楚再去掉
```

读哪几列(2026-10-07 在生产数据上只读实测:`daily_capital/` 下 `20180102.parquet` 到
`20260916.parquet` 共 2114 个文件,**每一个**都恰好是
`date, symbol, day_status, circulating_capital, total_capital, circulating_market_value, total_market_value`
这七列,symbol 写法如 `000001.SZ`):

| 字段 | 读的列 | 含义 |
|---|---|---|
| `FloatCap` | `circulating_market_value` | 流通市值 |
| `TotalCap` | `total_market_value` | 总市值 |
| `FloatShares` | `circulating_capital` | 流通**股数**(`SHARE_COLUMN_CANDIDATES` 的第一个) |

`total_capital` 是总股数,没有对应的存储字段(没有 `TotalShares`),不读。例:000001.SZ 在
2026-09-16 的 `circulating_capital` 是 1.940568e10 股、`circulating_market_value` 是 2.270465e11,
约 11.70 元/股。`SHARE_COLUMN_CANDIDATES` 里其余的拼写(`circulating_shares`、`float_shares`
……)只是给别的 schema 留的退路,在生产 schema 上一个都匹配不上;一个候选都不匹配时
`FloatShares` 就不建(从不反算)。每个字段实际读了哪一列记在 `meta.json` 的 `columns_used` 里。

脚本做的事,按顺序:

1. **只读**打开 cube store,只为拿它那套 canonical symbols / dates —— 平面天生就是对齐的;
2. `--out` 落在 `--cap-dir` 或 store 目录里面就**拒绝运行**,源数据目录永不写入;
3. 先探一个 parquet 的 schema 并打印列名,然后**把 store 的前 10 个 symbol 和 parquet 的
   前 10 个 `symbol` 并排打出来**;
4. 每个日期读一次 parquet(流式,内存里从不超过一天的 frame),拼成 `(S, D)` float64;
5. 写进临时目录再 `os.replace` 换过去 —— **半成品目录永远不可读**;
6. 任何字段的有限覆盖率低于 `--min-coverage` 就**非零退出**,并打印逐字段数字;
7. 最后打印 derived / log 字段名,并注明它们不需要建库(省得有人去找永远不会存在的文件)。

第 3 步和第 6 步是**唯一**挡得住"符号格式不匹配把每个格子都 NaN 掉"的防线,
而那个失败模式不报错、只是悄悄产出一个空因子。`scripts/tearsheet.py` 的 `load_caps`
正是把 `df["symbol"]` 直接 `.map` 到 store 的符号表上,而且**零测试覆盖**。

**不会**从 `FloatCap / Close` 反算 `FloatShares`:`Close` 是后复权的,比值不是股数。
**不会**写 `Log*` 平面:那是 `dailyfields.resolve` 在 float64 里现derive的。

### 1.5 数据身份:两个 hash

- **表达式身份**:`expr.key()` 带 `DF:` 前缀(`DF:FloatCap`),所以引擎的 CSE、GP 的
  种群去重 / 封禁查找、种子去重全都天然把它和同名分钟字段 `F:` 区分开。
- **数据身份**:`store.contract_hash` 在**两个** store 类里都混入了
  `dailyfields.fingerprint(self.daily)` —— **仅当真的挂了日频库时**,所以没挂库的
  store,hash 与改动前逐 byte 相同。`SegmentedCubeStore` 不是 `CubeStore` 的子类、
  也不调 `super().__init__`,两处都得各自改,别只改一处。
- **exemplar 缓存身份**:`rankban.view_cache_key(..., daily=None)`。这个 key 以前
  **完全不含 store 身份**,而 `ExemplarCache` 把排名指纹按 `root/<view_key>/` 存盘、
  构造时还会删掉别的 view key —— 日频库重建一次,旧指纹会活下来,排名封禁就会去拒错的候选。

---

## 2. 四个生成器关键字

`TypedTreeGenerator.__init__` 新增(全部可选,默认等于今天的行为):

```python
op_weight:    Mapping[str, float] | Callable[[OpSpec], float] | None = None
accept:       Callable[[Expression], bool] | None = None
daily_fields: Sequence[str] = ()        # 搜索可以抽的日频叶子
daily_leaf_prob: float = 0.0            # 深度 1 的 daily 槽位抽到日频叶子的概率
leaf_slots:   Mapping[str, Mapping[int, Sequence[str]]] | None = None
# 另外两个:
on_exhausted: str = "fallback"          # 或 "raise"
mutation_checks_bans: bool = True       # 只为逐 bit 复现某次老的 --ban run 才设 False
```

- **`op_weight`** —— 算子权重。`None` 时 `_pick` 走的是改动前**原封不动**那两行
  (`self._rng.choice(faces[self._rng.choice(sorted(faces))])`),两次 `choice`、
  同样的顺序。这是整个改动里最要紧的一条保证:加权路径用 `choices`,消耗的是
  `random()` 浮点数,两条路**不可互换**(实测 `Random(7)`:两次 `choice` 后
  `random()` = 0.3948…,两次 `choices` 后 = 0.6509…),一旦走错,第一代就全盘发散。
  加权路径是**按面平衡**的:一个面的质量 = 它那些算子权重的**均值**,面内再按各自权重抽。
  **权重 0 就是"把这个算子从词表里去掉、又不挪动抽样次数"的写法。** 权重全 0 的面会掉出去;
  要是把所有面都掏空,自动退回均匀路径 —— 权重函数不许把生成搞成不可能。
  `mutate_operator` 本来就走 `self._pick(alts)`,所以权重自动也管算子替换。
  **`{名字: 权重}` 这种写法是按 `spec.name` 匹配的,而每个 nd 面核函数在注册表里注册了
  两次** —— ND 面用自己的名字,NT 面用 `nt_<name>`。**映射的 key 自动展开到孪生体**:
  `{"cs_mean": 0.0}` 把 `cs_mean` 和 `nt_cs_mean` 一起掐掉,不必手写两遍
  (`generator.nd_twin_base`)。要**区别对待**两个面,就显式给 `nt_` 那个名字一条自己的
  权重 —— 显式写的永远优先:`{"cs_mean": 0.0, "nt_cs_mean": 1.0}`。
  传**可调用对象**时不展开:它自己看得见每个 spec 的两个面,怎么匹配是调用方的事。
  打错的算子名在构造时报 `ValueError`。
  (展开是分支验证后补的。改之前只写一半就只指引一半:实测 400 个 seed ×
  `generate_population(25)`,`{"cs_mean": 0.0}` 让 `cs_mean` 从 148 次降到 0,而
  `nt_cs_mean` 不动,17 次还是 17 次。验证报告里"5 次升到 9 次"是 120 个 seed 下的噪声,
  不是质量被挪给了孪生体 —— `_pick_weighted` 取的是**存活**算子权重的均值,掐掉一个
  权重 0 的算子并不改那个面自己的概率。)
- **`accept`** —— 候选筛子,签名**只能有一个位置参数**(构造时检查)。它挂在封禁集
  所在的那个验收点上,所以 `generate` / `mutate_subtree` / `mutate_operator` /
  `mutate_hoist` / `crossover` 全都免费生效。
- **`daily_fields` + `daily_leaf_prob`** —— 让搜索**抽得出**日频叶子(不只是能解析)。
  `self.daily_fields and ...` 的短路守卫意味着空默认**一次随机数都不消耗**。
  字段名在**构造时**就对着 `registry.daily_fields` 查一遍,`leaf_slots` 里的名字同样查
  (`generator.check_daily_field_names`)。少一个字母不会悄悄生效:
  `daily_fields=("FlotCap",)` 直接 `ValueError` 并列出九个可用名字。以前不查 ——
  `DailyFeature` 的 shape 只看节点类型、不看字段名,打错的叶子一路通过 shape 系统和
  `is_valid_root`,到求值时才被 `dailyfields.resolve` 拒绝,而 `GPMiner._panels` 是
  `tolerant=True`,把异常收进 `res.errors` 就把那棵树丢了(实测 `daily_leaf_prob=1.0`
  时 40 棵里 34 棵带着这个叶子,整轮看起来就像"日频叶子这条指引没起作用")。
  **"名字合法"和"这个 run 有数据"是两件事**:没挂 `--daily-fields` 的视图只解析得出
  `LimitWidth` / `BoardCode` / `ListAge` 三个派生字段,`FloatCap` 这种带 hooks 写进去
  照样构造成功、照样抽得出来、求值时照样被丢掉 —— 所以 `GPMiner` 会 `warnings.warn`
  点名这些字段(也留在 `miner.unresolvable_daily_fields` 上),`run_mining` 会
  `say()` 一行 `!! search hooks name N daily field(s) this view cannot resolve: ...`
  并写进 `meta.json` 的 `unresolvable_daily_fields`。想当场拦住就自己比一下
  `view.daily_fields`。
- **`leaf_slots`** —— 位置提示,`_operand_reqs` 表达不了的那种:一个 nd 面二元算子的
  **两个**槽位都是 `"daily"` 要求,所以搜索本身无从知道哪个槽位是"分桶用的条件变量"。
  `leaf_slots={"cs_regres": {1: ("FloatCap",)}}` 让 `_build` 直接把槽位 1 填成日频叶子
  而不是递归下去 —— 这才把"在自己的市值分位里排序"变成搜索**抽得出来**的东西。
  key 同样按 `spec.name` 匹配,也**同样自动展开到 `nt_` 孪生体**(查表在用时做,
  `_slots_for`,所以 `gen.leaf_slots` 还是你传进去的那个对象,插件按块改它仍然生效)。
  要让孪生体不受这条槽位约束,给它一条**空**的:`{"nt_cs_regres": {}}`。
  注意 NT 面的操作数必须有真的分钟变化,日频叶子塞进去会 `ShapeError`、`_gen` 重试,
  所以孪生体继承槽位的效果是**把它从种群里筛掉**,而不是填上什么 —— 这跟以前手写两个
  key 的效果一样(实测:只写 `cs_regres` 时 40 个 seed 里有 1 棵 `nt_cs_regres` 的槽位 1
  没被填;自动展开后、或两个 key 都写时都是 0;写 `{"nt_cs_regres": {}}` 回到 1)。

**`hook_rng`**:`accept` 钩子要抽随机数,**必须**用 `gen.hook_rng`,**不许**用 `gen._rng`。
用了 `_rng`,轨迹就取决于被筛掉了多少个候选,steer / control 两臂的对比也就毁了。
`hook_rng` 的种子是从生成器自己那条流的 **state** 派生的(`getstate()` 不消耗抽样,
所以树的随机流仍逐 bit 不变)—— 不是设计文档里写的那个字面公式,原因见 §6.8。

### 2.1 最小可运行例子

```python
from alphacube.catalog import default_registry
from alphacube.mining.generator import TypedTreeGenerator

gen = TypedTreeGenerator(
    default_registry(), seed=7,
    # nd 面的名字自动连 nt_ 孪生体一起生效,只写一个就够;day_sum 是 dt 面、只注册一次
    op_weight={"day_sum": 10.0, "day_std": 0.0, "cs_mean": 0.0},
    accept=lambda e: "cross_" not in repr(e),           # 只能有一个位置参数
    daily_fields=("FloatCap", "LogFloatCap"), daily_leaf_prob=0.3,
    leaf_slots={"cs_regres": {1: ("FloatCap",)}},       # 同样自动覆盖 nt_cs_regres
)
pop = gen.generate_population(30)
any("day_std" in repr(e) for e in pop)      # False —— 权重 0 生效
any("cs_mean(" in repr(e) for e in pop)     # False —— ND/NT 两个面都掐掉了
any("FloatCap" in repr(e) for e in pop)     # True  —— 抽到了日频叶子
any("cross_" in repr(e) for e in pop)       # False —— 筛子生效
gen.unaccepted_fallbacks                    # 0 —— 没有走退化兜底
```

构造时就会报错的五种配置:`op_weight` 里有没注册过的算子名;权重是负数或 NaN;
`accept` 要两个参数;`daily_fields` / `leaf_slots` 里有不在 `registry.daily_fields`
里的字段名;权重表把那 26 个 `reduces_m and not non_root` 的算子池掏空了
(掏空会让兜底路径走到 `self._rng.choice(sorted({}))` → IndexError)。

### 2.2 三处**故意**的行为改变

这三条不是开关后面的,是无条件生效的 —— 但对"不封禁任何东西、不装钩子"的 run 依然逐 bit 相同。

**(a) `mutate_window` / `mutate_constant` 现在也查封禁集和钩子。** 它们以前结尾是
`return child if is_valid_root(child) else None`,绕过了封禁集。实测漏法:封掉
`O:day_argmax(F:Low)` 之后,`mutate_window(cross_quantile_q25(day_argmax(Low), 3))`
照样返回 `cross_quantile_q25(day_argmax(Low), 15)`。`tests/test_mining.py:364` 的
docstring 本来就断言了这条不变式("封禁必须在每个验收点都成立"),它今天能过
只是因为它恰好挑中的载体 `cs_scale(day_first(Vwap))` 既没有窗口节点也没有常数。
**用 `--ban` 或滚动块的 run,轨迹会与以前不同**;要逐 bit 复现某次历史 run,
传 `mutation_checks_bans=False`。

**(b) `generate()` 的兜底路径现在也尊重钩子和封禁集。** 以前它**完全**绕过
`acceptable()`:实测一个恒假的 `acceptable`,`generate()` 照样返回
`day_cov(Close, Volume)`,`generate_population(8)` 返回 8 棵没过验收的树。
对 Partura 这种 steer / control 对照来说,这正是"指引越紧、种群里没被指引过的比例越大"。
现在:没有封禁也没有钩子时循环只跑**一次**、结果逐 bit 相同;有钩子时给钩子 64 次机会,
全被拒就返回一棵**尊重封禁集**的树并把 `unaccepted_fallbacks` 计上一次;
封禁集谁都活不下来则抛 `GenerationExhausted`(那是配置错误)。`on_exhausted="raise"`
是严格变体。

**(c) `--fitness` 去掉了 argparse 的 `choices=`。** 打错现在由 `cmd_mine` 对着
`scorer_kinds()` 拒绝(报错信息里列全),不再由 argparse 拒绝;shell 补全也不再提示列表。
原因:`build_parser()` 在插件加载**之前**跑,静态 `choices` 看不见插件注册的 kind。

---

## 3. 打分器注册表

```python
from alphacube.mining.scoring import register_scorer, scorer_kinds

register_scorer("my_kind", my_factory, replace=False)
scorer_kinds()        # ('ic', 'rank_ic', 'long', 'long_hit', 'my_kind')
```

`factory` 会被 `make_scorer` 这样调用:

```python
factory(target, dates=<view.dates 或 None>, warmup=<dates 里的预热行数>,
        turnover_penalty=..., signed=..., cost_per_side=..., top_frac=...,
        min_coverage=..., min_scored_days=...)
```

返回值要满足的协议**不是新发明的**,是现有代码已经在按鸭子类型用的那四样:

| 成员 | 谁在读 |
|---|---|
| `.score(panel, gated=True) -> dict`,含 `"fit"`(有限数;`-inf` = 这个候选被拒) | `gp.py` |
| `.score_combined(panel) -> dict`,含 `"fit"` | `env.py`(RL 奖励路径) |
| `.target_t64`、`.target_ranks_t` | `GPMiner` 两个都读 |
| `.columns`(可选) | 存档里多出来的列名 |

**`dates` 里是**包含**预热行的**,`dates[warmup]` 才是第一个真正被打分的日子。
`make_scorer` 自己不做任何切分 —— factory 要靠这两个参数自己切 train / valid。
这件事必须说清楚:`pipeline.py` 是 `view, trim = warmed_view(...)`,而 trim 只在**报告**
时才应用,所以 `view.dates` 带着 60 行预热;拿一个裸 `dates` 去切会把切分点悄悄挪位。

重复注册同一个 `factory` 对象是**幂等空操作**(插件加载器每次都会再跑一遍 `register()`);
撞上内置名字(`ic` / `rank_ic` / `long` / `long_hit`)一律报错;撞上已注册的名字而
factory 是**另一个**对象时,除非 `replace=True` 也报错。

### 3.1 最小可运行例子

```python
import numpy as np
from alphacube.mining.scoring import PanelScorer, make_scorer, register_scorer, scorer_kinds

class MyScorer:
    def __init__(self, inner):
        self._inner = inner
        self.target_t64 = inner.target_t64
        self.target_ranks_t = inner.target_ranks_t
        self.columns = ("my_extra",)
    def score(self, panel, gated=True):
        return {**self._inner.score(panel, gated=gated), "my_extra": 1.0}
    def score_combined(self, panel):
        return self._inner.score_combined(panel)

def my_factory(target, *, dates=None, warmup=0, **kw):
    # dates[warmup] 是第一个被打分的日子 —— 自己在这里切 train / valid
    return MyScorer(PanelScorer(target, kind="rank_ic"))

register_scorer("mine_demo", my_factory)
sc = make_scorer(np.random.default_rng(0).normal(size=(40, 12)), "mine_demo",
                 dates=[f"2020{i:04d}" for i in range(40)], warmup=5)
sc.score(np.random.default_rng(1).normal(size=(40, 12)))["fit"]
```

挖掘时用:`--fitness mine_demo`,或库调用 `gp_params={"fitness_kind": "mine_demo"}`。

---

## 4. 插件加载

**AlphaCube 永不主动 import 任何插件,源码里也从不提某个插件的名字。** 是调用方点名的:
`--plugin partura.ac_plugin`(可重复)、`$ALPHACUBE_PLUGINS`(逗号分隔,只在完全没给
`--plugin` 时生效),或者库调用 `alphacube.plugins.load_plugins([...])`。

一个插件模块可以有两个约定函数:

```python
# my_plugin.py
from alphacube.operators.common import op
from alphacube.plugins import SearchHooks, declare_real_valued
from alphacube.mining.scoring import register_scorer

@op("dt", "day", arity=1, real_valued=True)       # import 时就落进 OPERATOR_REGISTRY
def day_demo_range(x):
    """``day`` 族的约定:对分钟轴归约后 unsqueeze(-1),把宽度留成 1。"""
    import torch
    return (torch.nanquantile(x.float(), 1.0, dim=-1)
            - torch.nanquantile(x.float(), 0.0, dim=-1)).unsqueeze(-1)

def register() -> None:
    """注册表**建起来之前**必须发生的事都放这儿。"""
    declare_real_valued("in_sustain_mean")        # 见 §5
    register_scorer("demo", _factory)

def search_hooks() -> SearchHooks:
    """只有 mine 会读。`names` 里**最后**一个暴露它的模块赢。"""
    return SearchHooks(op_weight={"day_std": 0.0},
                       daily_fields=("FloatCap",), daily_leaf_prob=0.2)

def _factory(target, *, dates=None, warmup=0, **kw):
    from alphacube.mining.scoring import PanelScorer
    return PanelScorer(target, kind="rank_ic")
```

```python
from alphacube.plugins import load_plugins, search_hooks_from
from alphacube.catalog import default_registry

load_plugins(["my_plugin"])          # -> ['my_plugin']
reg = default_registry()             # 244 -> 245,新算子在里面
search_hooks_from(["my_plugin"])     # -> SearchHooks(...)
```

```bash
~/.venvs/alphacube/bin/python -m alphacube.cli mine --store /tmp/demo/cube2 \
    --daily-fields /tmp/demo/daily_fields --plugin my_plugin --fitness demo \
    --label close --budget 24 --population 12 --out /tmp/demo/runs --run-id m1
```

**同一份 `SearchHooks` 对 GP 和 RL 都生效**(`--miner alphagen` / `alphaqcm` /
`alphaforge` / `alphasage` / `warmstart` 这五个都算 RL),所以一个 sweep 里拿同一份
steer 配置分别指 GP 臂和 RL 臂,两臂是**可比**的:

| `SearchHooks` 字段 | GP 这条路 | RL 这条路 |
|---|---|---|
| `op_weight` | `TypedTreeGenerator._pick_weighted` | 折成每个 `OperatorToken` 一个 `log(max(w, tiny))` 的 logit 偏置,在 `action_mask` **之前**加到策略分数上(`tokens.operator_logit_bias` → `TokenMiner(logit_bias=...)`),所以权重 0 两边都等于抽不到 |
| `accept` | `generator.acceptable` | `BatchRewarder(accept=...)` |
| `daily_fields` | 生成器的日频叶子 | `Vocabulary(daily_fields=...)` 追加的 `DailyFeatureToken`(**追加在最后**,存盘策略的下标不动) |
| `daily_leaf_prob` / `leaf_slots` | 生效 | 不适用(RPN 词表里没有"槽位"这个概念,策略自己决定下一个 token) |

`mine` 的日志里这两行就是接上了的证据(没有 hooks 时两行都不打印):

```
RL action space: + 2 daily-feature token(s) (FloatCap, LogFloatCap)
RL op_weight -> logit bias: 5 of 214 operators down-weighted, 0 up; lowest -20.72
```

> QCM 是按**价值**选动作的,`log(w)` 不是它那条分支的量纲:`w=0` 时 `log(1e-9) = -20.7`
> 对着 `|rank IC|` 量级的分数等于一个硬掩码(这是要的性质),但 0 和 1 之间的权重只是
> "被压制",不是校准过的倍数。想要概率意义上的加权,用三个 softmax miner
> (`alphagen` / `alphaforge` / `alphasage`)。

机制与时序:

- 光 **import** 就够用来注册 `@op` 装饰的核函数(它们落进 `operators.common.OPERATOR_REGISTRY`);
  `register()` 是给"必须在注册表建起来**之前**发生"的事准备的 —— `declare_real_valued()`、
  `register_scorer()`,以及往 `operators.operator_dict_DT` / `_ND` 里塞东西
  (那是让 `fullcatalog` / `parse_legacy` 的名字表也看见新核函数的途径)。
- `load_plugins` 最后会把 `catalog._DEFAULT = None`,下一次 `default_registry()` 就会
  带上新算子重建。**注意**:一个库调用方如果**已经**拿着某个 `Registry` 对象,它看到的
  还是短词表,症状是 `ParseError: unknown operator`,读起来像打错字。
- `cli.main()` 的时序是实测安全的:`build_parser()` 返回后 `catalog._DEFAULT` 仍是
  `None`,而 `alphacube.operators` 已经 import 过(所以 `@op` 装饰器已经落地)。
- `search_hooks_from` 和 `load_plugins` 都靠 `importlib.import_module` 的缓存,
  **可以重复调用、顺序随意**。

---

## 5. `real_valued`:布尔探针会误判

注册表靠**探针**判断一个核函数的输出是不是布尔 / 平凡 / 粗粒度(不手写名单,这是项目的
既有约定)。但有两类核函数会被误判:**只在精确相等上触发的测试**(连续随机浮点数上永不触发),
以及**分位分桶**(只有 9 列探针数据时 `n >= 2` 的闸门永不触发)。被误判的算子会被
`searchable(include_boolean=False)` 排除掉,而且不能当根。

两个互补的声明通道,都**默认空 / 关**:

```python
@op("dt", "day", arity=1, real_valued=True)      # 定义点声明,给自己的核函数
def my_kernel(x): ...

from alphacube.plugins import declare_real_valued
declare_real_valued("in_sustain_mean")           # 按名字声明,给不属于自己的核函数
                                                 # (实际改 catalog.REAL_VALUED_OVERRIDES)
```

`_make_spec` 读到声明后是**跳过**探针,不是覆盖探针的结论
(`distinct = 10**9`、`trivial = False`)。每个探针都用自己本地 seed 的
`torch.Generator()`,所以为某个核函数跳过探针,不可能扰动别的核函数的判定、
也不可能扰动搜索自己的随机流。一次声明同时拨动两个闸门:
重新进入 `searchable(include_boolean=False)`,**并且**变成合法的根。

**当前实测数字**(别再重新推测):

| | 值 |
|---|---|
| `len(registry)` | 244 |
| faces | `dt=121 nd=48 nt=48 shared=27` |
| `boolean_output` / `trivial_boolean` / `coarse_output` | 29 / 22 / 0 |
| `searchable(include_boolean=False)` | **218** |
| 生成器实际默认(`include_boolean=False, include_resource_risk=False`) | **214** |

AlphaCube 自己的核函数里**恰好三个**被误判:`in_sustain_mean`、`in_sustain_stddev`、
`in_sustain_product`。把这三个声明掉会让词表变成 **221 / 217**(实测)。
**本次改动把这个机制做出来了,但一个都没声明** —— 所以不加 `--plugin` 的 run 仍是 214,
没有任何 GP 轨迹会动。那三个的声明放在 Partura 插件里。
`in_sustain` 本身必须保持被标记:它确实在传播一个 0/1 掩码。

`tests/test_real_valued_flag.py` 把那 29 / 22 / 0 三个集合**逐名字钉死**了 ——
这是今天整个测试套件里没有的东西,也正是这两类误判一直沉默的原因。

---

## 6. 读的时候别踩的坑

1. **`(S, D)` 还是 `(D, S)`。** `view.daily(name)` 返回 **`(S, D)`** —— 引擎内部的面板
   方向,**不是** `EvalResult` 对外暴露的、也不是 `tradability.board_limit()` /
   `tearsheet.load_caps()` 递回来的那个 `(D, S)`。那**一次**转置发生在
   `dailyfields.derive()` 的 `LimitWidth` 分支里,别再转第二次。
2. **`LimitWidth` / `BoardCode` 依赖符号格式。** 实测:`SH.600000` 这种写法下
   **每条板块规则都不命中**,`board_limit` 返回一个均匀的 0.10 平面、覆盖率 100%,
   于是没有任何覆盖率下限能发现它。`derive()` 因此会在**没有任何**
   STAR(688/689) / GEM(300/301/302) / BSE(4/8/92) 规则命中时 **抛 `ValueError`**,
   并把前五个符号和认得的格式都报出来。合成 store 的符号写法也因此从 `SH.6xxxxx`
   改成了 `6xxxxx.SH` 混搭(含一个 `688xxx.SH`、一个 `300xxx.SZ`、一个 `8xxxxx.BJ`),
   这就是为什么 `synthetic.py` 和 `conftest.py` 必须同一个包改。
3. **`day_sum(log(FloatCap))` 在 legacy 下是接受的,而且等于恒等。** 这是**钉住、没修**:
   宽度 1 的历史豁免放它过去。`strict=True` 会拒。`day_sum(div(FloatCap, 2.0))`
   **两种模式都拒**(`div(DF, Constant)` 是 WIDE_DEGENERATE,不是 collapsed)。
   有测试钉着,所以将来谁改那个豁免,不会悄悄改变"日频叶子是什么意思"。
4. **预热期悄悄为空。** `resolve` 会在某个字段的第一个有限日期晚于 `dates[0]` 时
   **每个字段警告一次**。看见这个警告别忽略:那意味着预热窗里那个字段全是 NaN。
5. **`ListAge` 锚在 store 的日轴上**,不是锚在当前窗口。否则同一个 (symbol, date) 格子
   在全量 view 上值 22.0、在往后挪 20 天的 view 上值 10010.0,而且因为窗口首日之前上市的
   股票全都领到哨兵值,平面会在横截面上**退化成常数** —— `cs_rank(ListAge)` 变成一片并列。
   `contract_hash` 不随窗口变,所以没有任何东西会标记这件事。
6. **指引 nd 算子时,`nt_` 孪生体要单独写一遍。** `op_weight` 的 `{名字: 权重}` 形式和
   `leaf_slots` 的 key 都按 `spec.name` 匹配,而 48 个 nd 面核函数每个都注册**两次**
   (ND 面用自己的名字,NT 面用 `nt_<name>`)。所以 `{"cs_mean": 0.0}` 只掐掉一半,
   质量还会**挪到**孪生体上(实测 120 seed:`nt_cs_mean` 从 5 次升到 9 次)。
   这不会报错 —— 打错名字会报,"只写了一半"不会。Partura 侧已经有
   `ac_plugin.partura_operator_names()` 专门把 `nt_` 展开,写新的指引时照抄那个做法。
7. **日频叶子名字打错不会在构造时报错。** `op_weight` 的算子名在构造时就校验,
   但 `daily_fields=` / `leaf_slots=` 里的**字段名没有校验**。GP 用
   `tolerant=True` 求值,所以 `daily_fields=("FlotCap",)` 这种打错会让每个带它的候选
   **静默地**在求值时被丢掉,看起来就像指引没起作用。用 `view.daily_fields` 或
   `dailyfields.available(store.daily)` 先对一遍名字。
8. **`hook_rng` 的种子不是设计文档里那个字面公式。** 设计写的是
   `random.Random((seed * 2654435761) & 0xFFFFFFFF)`;实现改成了从生成器那条流的
   `getstate()` 派生。原因实测:唯一的在树内构造点 `gp.py` 是用 `rng=self._rng` 建生成器、
   **不传 `seed=`** 的,所以 `seed` 一直停在默认的 0,每一次挖掘(seed 0 / 7 / 12345 一样)
   拿到的都是同一条钩子流 `[0.844422, 0.757954, 0.420572]`;而 `seed=0` 时那条流
   **逐 bit 就是 `self._rng` 本身**,于是一个在 `hook_rng.random()` 上做筛选的钩子
   读的其实是生成器自己的抽样。`getstate()` 不消耗抽样,所以树的随机流仍逐 bit 不变。
9. **`cs_bucket_rank` 不是 AlphaCube 的算子**,是 Partura 的(`partura/ops_ext.py`)。
   设计文档的若干例子拿它举例,在不加 `--plugin` 的裸 AlphaCube 上会
   `ParseError: unknown operator`。本文 §2.1 的 `leaf_slots` 例子用的是基础词表里真有的
   `cs_regres`。

---

## 7. 向后兼容:为什么默认什么都不变

逐条列在设计文档的 "Backward compatibility"(15 条)。要点:

- 新叶子类 `DailyFeature` 是**新类**,不是 `Feature(field, daily=True)` ——
  别处有五个 `isinstance(n, Feature)` 的判断点,加 flag 会让它们全部悄悄改变语义。
- 9 个日频字段名和 20 个别名与 `CUBE_FIELDS` / `FIELD_ALIASES` / `DAILY_MACROS` /
  `DAILY_NAME_ALIASES` / 244 个算子名 / 完整 catalog **两两不相交**(有测试断言),
  所以**没有任何**已有的表达式字符串能包含它们,也就没有任何已有的树能含这个节点。
- `Registry.fields` 仍**恰好**是那 8 个分钟字段,日频字段放在**并行**的
  `registry.daily_fields` 里 —— 这才是整个日频特性对搜索随机流不可见的原因
  (`TypedTreeGenerator.fields` 和 `Vocabulary.fields` 都默认取 `registry.fields`)。
- `Vocabulary(registry)` 的 token 下标逐 bit 不变,存盘的策略照样能载入;
  日频 token 只在传了 `daily_fields=` 时**追加在最后**。
- 没挂日频库时,两个 store 类的 `contract_hash`、`view_cache_key` 的哈希串都**逐 byte 相同**。

---

## 8. 测试

```bash
cd /Users/wxy/Desktop/AlphaCube-steering
~/.venvs/alphacube/bin/python -m pytest tests/ -q 2>&1 | tail -8
# 943 passed, 10 skipped, 1 xfailed —— 改动前的基线是 617 passed, 10 skipped, 1 xfailed
```

仓库根的 `conftest.py` 会把**当前 worktree 自己的 `src/`** 放进路径,所以在哪个 worktree 里
跑就测哪份代码。Partura 侧(`partura/ac_plugin.py` 等)要**显式**指过来,否则会抓到
别的 AlphaCube 检出:

```bash
cd /Users/wxy/Desktop/Partura-steering
ALPHACUBE_ROOT=/Users/wxy/Desktop/AlphaCube-steering \
PYTHONPATH=/Users/wxy/Desktop/AlphaCube-steering/src:/Users/wxy/Desktop/Partura-steering \
  ~/.venvs/alphacube/bin/python -m pytest tests/ -q        # 57 passed
```

新增的测试文件,按工作包:

| 文件 | 覆盖 |
|---|---|
| `tests/test_real_valued_flag.py` | 29 / 22 / 0 三个集合逐名字钉死;218 / 214;声明机制两条通道 |
| `tests/test_daily_leaf_typing.py` | shape 恒等、`DF:` key、语法表、两种模式的拒绝、别名与不相交性、`derive()` |
| `tests/test_daily_fields_store.py` | 按**标签**对齐(超集 / 子集 / 乱序 / 缺日期)、每 view 只算一次、`contract_hash` |
| `tests/test_daily_leaf_engine.py` | 三个叶子读取点、分块不变性、两个结构 key 读取点 |
| `tests/test_generator_hooks.py` | **随机流恒等**(两种查法)、权重、构造校验、兜底、两个漏洞 |
| `tests/test_scorer_registry.py` | 空注册表下内置分支不变、factory 收到的 `dates`/`warmup`、重名 |
| `tests/test_plugin_hooks.py` | `hooks=None` 与不传等价、插件加载、墙钟、`DF:` 封禁键、缓存身份 |
| `tests/test_cli_daily_and_plugin.py` | `mine`/`research`/`calculate` 端到端、`--fitness` 校验、RL 侧校验 |
| `tests/test_build_daily_fields.py` | 手写 parquet 目录、覆盖率下限、`--out` 拒绝、半成品不可读 |
| `tests/test_daily_tokens.py` | token 下标不变、`ExpressionBuilder`、权重 0 的 logit 偏置 |
| `tests/test_steering_wiring.py` | 分支验证的五条回归:logit 偏置真的接到五个 RL miner 上、`hooks.daily_fields` 真的进词表、解析不了的日频字段被点名、ND/NT 孪生体、日频字段名校验 |
