# AlphaCube 交接

你正在接手 AlphaCube。这份文件是自足的 —— 读完就能接着干,不需要任何对话历史。
设计层面的细节在 `docs/design.md`(中文,权威);这份讲**在做什么、怎么操作、踩过什么坑**。

---

## 0. 现在的状态(2026-09-19 00:40)

**一个全市场挖掘任务正在跑**,别动它,除非你有理由停。

```
run-id   gpu8              screen 会话 gpu8(任务)+ gpu8rep(每 5 分钟快照 -> gpu8_report.log,带 anon 内存)
pid      313998            00:34 启动(scripts/gpu8.sh)
配置     20 代 = 4 块 x 5 代,全市场 5220 只 x 20180102~20251231,seed 0
         池子上限 30、块结算前 120 条、最终入池前 60 条
         开局 --ban 9 个结构:gpu5 两块的 6 个 + gpu7 第 1 块的 nt_ts_kurt(Vwap)|w30、pow_1_5(Amt)、day_max(Amt)
         --ban-rank-corr 0.95:和被封结构的因子样本 |rank 相关| >= 0.95 的候选一律拒绝(用户定的口径)
         样本 = gpu5 第 10 代 + gpu7 第 7 代 archive 中含这 9 个键的 206 条(--ban-archive),启动时先求值
代码     commit 29e2559
```

**为什么有 rank 封禁**:gpu7 第 1 块证明按一层结构封禁能被单调改写绕过 —— `log(Amt)` 封了,
GP 写成 `log(pow_1_5(Amt))`、`pow_05(pow_05(day_max(Amt)))`,日截面排名与 `day_max(log(Amt))` 逐位相同。
rank 相关对任何单调改写(含反号)恰为 1.0。生成日志每代多 `similar`(本代被拒数)与 `exemplars`(样本数)。

**gpu7 已停**(用户批准,00:33,第 7 代后):第 1 块报告 —— 头部 10 条只是 2 个因子:日内 Vwap 峰度
(`nt_ts_kurt(Vwap, 30)` 族,IC 0.031~0.032,日换手 81~89%,扣费夏普 −9.5~−18.6)与换写法回来的 `day_max(Amt)`;
融合回测扣 15bp 后只有 equal 为正(+0.24),限换手 30% 时 equal/ic/icir/lstsq/ridge 约 1.0~1.25,树模型为负。

**内存:容器上限 90 GiB(`/sys/fs/cgroup/memory.max`),不是 `free` 显示的 754 GB。** `ps` 的 RSS 含 mmap 数据文件页
(可回收,gpu7 启动 5 分钟 RSS 71.9 GB 而 anon 只有 19.8 GB),判断要看 `memory.stat` 的 `anon`。
**gpu6(不设上限)在第 1 块结算中途静默死亡**(~12:40,无 traceback、无 core、cgroup oom_kill=0):
池子 145 条 + 组合层全尺寸堆叠,叠在 GP 自己的面板缓存上撑爆内存。单独进程重放同一结算能跑完
(`_bench/replay_block1.py`),所以是叠加的问题。上限 30 时结算内存有界,与 archive 大小无关。

gpu6 重放出的第 1 块(不设上限,145 条,**扣 15bp**):ridge 夏普 5.37 → −0.13,gbdt 4.12 → −2.55,
lightgbm 3.74 → −2.83,xgboost 3.97 → −3.27(日换手 50~71%);equal/ic/icir/lstsq 全 NaN —— 组合层把
NaN 当传染(任一成员缺值整格 NaN),已修为缺值按 0(commit 7e50681)。

**gpu4 / gpu5 / gpu6 已停**:gpu5 跑到第 10 代(`mining_runs/gpu5/checkpoints/archive_gen001..010.csv`);
gpu4、gpu5、gpu6 前几代逐行相同。每代耗时 gpu4 75.1 / 22.3 / 24.7 / 15.5 分 → gpu5 11.4 / 8.4 / 12.5 / 5.2 分。
gpu5 第 10 代头部扣费回测见 `_bench/top_gpu5_gen010.json`(`_bench/eval_top.py`)。

**gpu5 的第 1 块结算(46 分钟)暴露了三个问题,均已修复:**
1. `MsePool` 每试加一条候选就在 ~800 万行上从头解最小二乘(k=31 时 22 秒,池满时每条解两次)
   → 改为增量 Gram 矩阵,单次插入 55.5 s → 5.1 s,成员与池 IC 不变(commit 52eba66)。
2. 融合回测**不扣成本**:`compare()` 不接成本参数,块报告和最终组合表只有无成本夏普,
   而模型类方法日换手 45~67%、简单加权 18~19% —— 恰是成本会翻转的排名(commit 7f723f9)。
3. `gbdt`(HistGradientBoosting)建模不带种子,早停随机划验证集,同输入同 seed 两次夏普 5.0 / 3.0
   (同一 commit)。另:滚块后那行日志的 `gen_best +nan ... valid 0` 是写死的占位值,不是故障
   (那批树实际进了 48 条 archive),已改为报真实数字(commit 5ca8ecf)。

gpu5 第 1 块报告(**无成本口径,样本外 = 后 30% 交易日,池子上限 30**):lightgbm ICIR 0.55 / 夏普 4.59 / 日换手 62%;
gbdt 0.54 / 4.34 / 61%;xgboost 0.54 / 3.89 / 67%;lstsq 0.42 / 4.44 / 54%;ridge 0.40 / 4.28 / 45%;
icir / ic / equal 约 0.27 / 3.4~3.6 / 18~19%。gpu7 的块报告带扣 15bp 与限换手两套。

第 3 代头部 10 条已按 pipeline 口径补算扣成本回测(`_bench/eval_top.py`,结果 `_bench/top_gen003.json`):
几乎全是**成交额水平**(低成交额 → 次日高收益,小盘/流动性溢价),能扛住 15bp 的只有低换手的平滑版
(`sub(0.5, abs_(ts_mean(day_max(log(Amt)), 10)))` 日换手 7%,扣费夏普 2.55,2018~2025 每年为正);
`day_durbin_watson` 族无成本为正、扣费后每年为负(日换手 44~46%)。全部是样本内。

`_bench/` 是这次会话的测量脚本目录(A/B、逐算子一致性、分块不变性),可删。

---

## 1. 这个项目是什么

三维立方体 `(股票, 交易日, 分钟)` 上的因子挖掘引擎。一个数据对象、一份算子注册表、
一个引擎,GP / 四个 RL miner / 池 / 组合 / 回测全跑在它上面。

**当前阶段**:在真实全市场数据上跑 GP,反复发现"搜索被钻空子"并堵漏 —— §7 那张表
里每一条都是 GP 在真实数据上找到并利用过的漏洞。

---

## 2. 位置与连接

| | |
|---|---|
| 本地 | `~/Desktop/AlphaCube`(git,remote = `github.com/xiangyu2024/AlphaCube`) |
| 本地 venv | `~/.venvs/alphacube/bin/python` |
| 服务器 | `ssh -p 29982 root@connect.westd.seetacloud.com`(**公钥已授权,免密**) |
| 服务器项目 | `/root/autodl-tmp/AlphaCube` |
| 服务器 python | `/root/miniconda3/envs/quant/bin/python` |
| 数据 | `/root/autodl-tmp/qmt_daily_data/fast_factor_compute`(5219 只 × 2111 天 × 241 分钟,79 GB) |

**连接务必带 `-o ControlPath=none`** —— 否则会去复用一个早已失效的 ControlMaster 套接字然后被拒:

```bash
ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com '<cmd>'
```

`ps -eo ... -p <pid>` 里的 `-e` 会**覆盖** `-p` 过滤而列出所有进程。查单个进程用 `ps -o pid=,etime= -p <pid>`。

### 两条硬约束

1. **只能改 `/root/autodl-tmp/AlphaCube`**,服务器上其他一切只读。装 pip 包、改系统设置都超出授权。
2. **系统盘 `/` 只有 30 GB 且已用 86%**(miniconda 22 GB + vscode-server 8.4 GB)。
   中间产物必须落在数据盘。引擎 `workdir` 默认已是 `<项目>/tmp`,**别改回系统临时目录** ——
   一个落盘边界文件是整块 `(S, D, M)` float32 = **9.8 GB**,曾经写满系统盘、进程被静默杀掉、
   日志里连一行报错都没有。

---

## 3. 跑一次挖掘

```bash
ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com \
  'cd /root/autodl-tmp/AlphaCube && for f in "" _report; do cp scripts/gpu8$f.sh scripts/gpu9$f.sh;
   sed -i s/gpu8/gpu9/g scripts/gpu9$f.sh; done &&
   screen -dmS gpu9 bash -c "./scripts/gpu9.sh > gpu9_run.log 2>&1" &&
   screen -dmS gpu9rep bash -c "./scripts/gpu9_report.sh"'
```

`scripts/gpu8.sh` 是当前配置的样板(只在服务器上,不在 git 里),改参数复用。**脚本里写死了
`--run-id`**:直接拿旧脚本起新任务会写进旧的 `mining_runs/<id>`,所以先复制再改名(上面那条命令就是)。
**长任务一律用 `screen`** —— 它与 ssh 连接无关,断网、关机、退出 IDE 都不影响。`nohup` 不够可靠,试过。
停任务用 `kill -INT <pid>`:KeyboardInterrupt 会走引擎的 `finally`,落盘临时目录会被清掉
(SIGKILL 不会,9.8 GB 一个的边界文件会留在 `tmp/`)。

### 当前配置的含义

```
--budget 1280 --population 64        20 代(budget / population = 代数)
--min-depth 2 --max-depth 5
--rolling-block 5 --ban-top-n 3      每 5 代推倒重来,封禁该块最常用的 3 个一层子结构
--cost-per-side 0.0015               单边 15bp,买卖都收,整本换一遍 = 2 倍
--turnover-cap 0.30                  另出一套"日换手率 ≤30%"的曲线
--min-coverage 0.5 --min-scored-days 0.5
--checkpoint-every 1                 每代把表达式 + 回测写盘
--progress-every 60                  代内进度(轮次 / units / 百分比 / ETA)
--panel-cache 1024                   面板缓存条数(一张 40 MB,1024 ≈ 42 GB)
--combine equal,ic,icir,lstsq,ridge,gbdt,lightgbm,xgboost
```

服务器上 lightgbm 4.6 / xgboost 3.2 / sklearn 1.8 都有;`factorgcl` 需要 `torch_geometric`,**未装**。

### 滚动块的语义

```
1~5 代   -> 融合回测(仅本块)         -> 封禁本块最常用的 3 个一层结构 -> 重建初始种群
6~10 代  -> 融合回测(仅本块 + 累计1~2) -> 再封 3 个(与前面的并集生效)  -> 重建
11~15 代 -> 融合回测(仅本块 + 累计1~3) -> ...
```

封禁是**累积并集**,且在**所有接受点**生效(新树、五种变异、交叉),被封结构无法通过繁殖绕回来。
最后一个不足 5 代的残块也会被评分。

---

## 4. 监控

```bash
S='ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com'

$S 'cd /root/autodl-tmp/AlphaCube && tail -20 gpu8_run.log'            # 进度
$S 'cd /root/autodl-tmp/AlphaCube && ls mining_runs/gpu8/checkpoints'  # 每代产物
$S 'ps -eo pid,etime,pcpu,rss,args | grep "[c]li mine"'                # 活没活着
$S 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'   # GPU
$S 'df -h / /root/autodl-tmp'                                          # 磁盘(系统盘必看)
```

**Claude Code 的 `Monitor` 工具里 ssh 连不上服务器**(2026-09-18 实测:两个 Monitor 各 30 分钟,一次也没连上,而同一条命令用后台 Bash 每次都通)。等服务器上的事件一律用 `Bash` + `run_in_background` + `until` 循环。

现成的查看脚本(都在 `scripts/`):

| 脚本 | 用途 |
|---|---|
| `show_gen.py <ckpt.csv>` | 某代头部因子:表达式 + IC/夏普/年化/回撤/覆盖/天数 |
| `show2.py <ckpt.csv>` | 同上但**按 \|夏普\| 排**(见 §6 的符号陷阱) |
| `degen.py <ckpt.csv>` | 某结构在 archive 里的占比 |
| `sharpe.py` | 逐代的 \|IC\| 最高 vs 夏普最高 |
| `combine_archive.py <ckpt.csv>` | 把任意 checkpoint 补跑一遍组合层(含 lightgbm/xgboost) |
| `diagnose_run.py --store ...` | **跑不出因子时逐层定位**:store→字段→label→闸→基础因子 |

---

## 5. 产物

```
mining_runs/<run-id>/
  checkpoints/archive_genNNN.csv   每代:rank, fitness, ic_mean, icir, rank_ic,
                                   coverage, n_days, annualized_return, sharpe,
                                   max_drawdown, expression(前 10 名带指标)
  members.csv                      最终池成员(全部带逐因子回测)
  combination.csv                  组合方法样本外对比
  combined.npy / values/*.npy      面板值
  meta.json                        配置与轴
```

**archive 只在 checkpoint 里。** GP 的 archive 活在内存,`_commit()` 只在预算跑完时执行。
中途杀进程、没有 checkpoint 就什么都拿不到 —— 这是花了一整夜换来的教训,所以
`--checkpoint-every 1` 基本应该一直开着。

---

## 6. 读结果时的陷阱(都踩过)

**符号**:GP 用 `|IC|` 做适应度,方向由组合层事后对齐。所以 `IC = -0.024, 夏普 = -3.3`
是一条**强**因子(翻转后夏普 +3.3),不是坏因子。真正可疑的是 **|IC| 高而 |夏普| 接近 0**。
按 `|夏普|` 排序(`show2.py`)才对应真实强度。

**回撤 -99%**:负 IC 因子被复利八年的算术必然(年化 -44% 复利 7.7 年 = 净值剩 1%),
不是"某天暴雷"。要看翻转后的。

**`coverage` 与 `n_days` 要一起看**:`coverage` 只问格点是否有限,一个恒为 0 的面板
处处有限、照样通过;`n_days` 问的是**当天有没有截面区分度**。两者背离就是退化因子。

**换手率**:每条因子都有 `turnover` / `*_net`(扣成本)/ `*_capped`(限换手)。
无成本夏普 4.0 配 89% 日换手,扣 15bp 后是负的。**永远看扣成本那套。**

---

## 7. 已经堵上的漏洞(别再打开)

搜索会钻一切空子。以下每条都是在真实数据上被 GP 找到并利用过的:

| 漏洞 | 现象 | 现在怎么防 |
|---|---|---|
| **平凡布尔算子** | `pow_05(eq(a,b))` 把恒定指示量洗成合法根,\|IC\| 0.58 而池里 IC 是 NaN | `trivial_boolean` 探针(喂全正/全负/混号/同操作数,任一使输出恒定即判定),22 个默认移出词表 |
| **float32 下溢** | `pow(0.5, 成交量)`:0.5^1e5 = 0,面板恒零,\|IC\| 0.0741 全靠 329/1893 天,且六代内霸占前 8 名 | 二元 `pow` 移出搜索,指数改为烘焙变体 `pow_05/1_5/2/2_5/3` |
| **部分退化面板** | 有值但每天截面无方差,IC 从少数天来(全退化的早被"IC 非有限"挡了,漏的是部分退化) | `--min-scored-days 0.5` |
| **NT 面配对核** | `nt_ts_pearson` 逐日 5219² 对 × 1942 天,一代跑 3~10 小时 | 按**面**排除:ND 的 `ts_pearson` 保留(整面板一次),NT 双胞胎排除(逐日 241 倍) |
| **单结构霸榜** | `day_ar1_coefficient(Volume)` 占 archive 的 49%,外面套不同单调变换,\|IC\| 全在噪声内 | `--rolling-block` + `--ban-top-n`;另加初始种群去重、变异从 3 种增至 5 种(含唯一能让树变短的 `hoist`) |

> 判据一律用**探针实测**而非手写名单 —— 见 `catalog._probe_constant_capable`。

**还修过一个静默 bug**:`pipeline.py` 里 `realized` 被重新绑定成截断版,而最终 `compare()`
传的是未截断的列和 target,导致每个组合方法都越界 —— 但异常被 `except` 捕获后**做成表格打印**,
所以跑完看着"正常结束",实际最重要那张表全是 `error`。现在有测试断言最终表必须有数字。

---

## 8. 还没解决的

| | 说明 |
|---|---|
| `sliding_{top,bottom}_{sum,mean}_masked_p5` | 声明为 `dt/in`(分钟轴)但 `_sliding_windows` 沿 **dim 0** 滚动,而引擎在该面喂进来的 `(A·B, M)` 里 dim 0 是**股票日** → 跨股票边界、结果随分块而变。已用 **strict xfail** 钉住(`test_sliding_masked_family_is_chunk_invariant`)。修法待定:改滚动轴 / 改注册到 ND 面 / 剔除 |
| 日频 `Vwap` 口径 | 新写法复现生产日频 store 到中位 8e-5,但不是原式,生产到底怎么算的未知 |
| `Return` 的 39 处 NaN 位置差异 | 与生产日频 store 比对时出现,约 0.8%,疑似复牌日,未细查 |
| ~~CUDA 逐算子一致性~~ | **已跑(2026-09-18,`parity_cuda.log`)**:240 个算子两边都跑通,0 个只在 CUDA 失败,没有算子在 CUDA 上可测地更差 |
| **CUDA 上分块不变性不成立** | CPU 上任意分块逐 bit 相同(测试锁定);**CUDA 上不是**:CUDA 按张量形状选归约路径,块的行数一变,`day_sum`/`day_mean` 有 10~15% 的格点差 ULP 级(相对 ~3e-7)。与行对齐无关(同一批行错开一个 float 再求和逐 bit 相同)。被**阈值**放大的核函数会翻 NaN:`day_durbin_watson` 用 `E[x²]-E[x]² < 1e-8` 判常数行,价格 ~10–100 时这个差的舍入噪声 ~1e-4,判定由噪声决定 —— 5 只一块对 15 只一块翻了 6~18 个格点,gpu4 第 3 代 #2/#5 的 IC 因此差 4.5e-5 / 4.5e-6。另外自动块大小取决于**同批所有树**的最大窗口,所以 GPU 上同一因子的值会随同批的其他树变。选择类(`day_max`)不受影响。修法待定:文档化 + 阈值核改两遍法方差(改 vendored 核,需用户拍板) |
| `factorgcl` 组合方法 | 需要 `torch_geometric`,服务器未装(装它要动 conda 环境,超出授权) |
| 引擎提速 | 读取已修(§9,同一批树 3.1 倍);下一步在核函数和每节点 D2H 同步上,尚未测量 |

---

## 9. 性能事实(全部实测,别再重新推测)

### 叶子读取曾是瓶颈 —— 根因是读法,不是 I/O,已修(2026-09-18)

下面 `_profile_gpu.py` 的测量没错,但"与设备无关的那段"不是磁盘也不是传输(整个 store 早在
页缓存里,buff/cache 490 GB),而是两处代码:

1. `SegmentedCubeStore.read_cube` 三个轴(连分钟轴)全走 `np.ix_`,逐元素 gather。生产块形状
   (15 只 × 1942 天,26.8 MB)实测 **58 ms**,同一块直接切片 **2.1–2.7 ms**(21–28 倍),逐 bit 相同。
2. 引擎每**用**一次叶子就重读一次。gpu4 第 2 代 archive 当种群:S 轮一块读 **104 次**,只有 **8 个字段**。

修法:读取按连续区间切片;一块之内每个操作数只物化一次、引用计数释放。服务器 GPU 上 A/B
(第 2 代 archive 129 棵树,20 只 × 1942 天,每块 5 只,新旧代码各跑一遍):

```
          总耗时   叶子读取+上卡              核函数(立方体)
旧        17.3s   11.65s (67%)  436 次 × 26.7 ms   4.69s
新         5.6s    0.30s ( 5%)   40 次 ×  7.6 ms   4.27s (76%)
```

129/129 面板逐 bit 相同。

**生产块形状上的实测**(`_bench/ab_modes.py`:第 3 代 archive 每 4 条取 1 条共 40 棵 ≈ 一代的新树数,
S 块 = 15 只 × 1942 天,D 块 = 5220 只 × 5 天,新旧代码各跑一遍,40/40 面板逐 bit 相同):

```
            每个块位置(所有同模式轮次合计)     × 生产块数      = 每代
S 轮   旧 2.23s(读 1.89s)  新 0.44s(读 0.10s)   5.0 倍  × 348   旧 13.0 分  新 2.6 分
D 轮   旧 0.98s(读 0.53s)  新 0.46s(读 0.05s)   2.2 倍  × 389   旧  6.4 分  新 3.0 分
```

旧代码的推算(S 13.0 + D 6.4 分)与 gpu4 第 2 代实际日志(S 轮约 13.5 分、D 轮约 6 分)吻合,
所以这个"块耗时 × 块数"的算法站得住(只放大块数,不外推代数)。引擎部分每代约 **3.5 倍**;
加上没变的 P 轮、适应度、checkpoint(约 1–2 分),一代从 ~22 分到 ~7 分,约 **3 倍**。
修复后 D 轮成了大头:NT 面逐日调核(设计上必须逐日,见 design.md §4)加上落盘边界的跨步读。

**下面"提速方向:预取/异步"作废** —— 读取只剩 5%,再藏也藏不出东西。
现在的大头是核函数本身,其次是"其余"(18%:调度 + 每个交接节点一次 D2H 同步)。

以下是修复前的测量,留作对照(242 天 × 400 只):

```
device=cuda   总 4.0s
  叶子读取+上卡    2.5s  (63.8%)   24 次,单次 106 ms   <-- 主因
  核函数(立方体)   0.4s  (11.3%)   27 次,单次  17 ms
  核函数(面板)    0.2s  ( 4.7%)
  其余(调度/落盘)  0.8s  (20.3%)

device=cpu    总 16.4s
  核函数(立方体)  12.9s  (79.1%)   单次 479 ms   <-- CPU 上核函数慢 29 倍
  叶子读取+上卡    2.4s  (14.8%)   单次 101 ms   <-- 与 CUDA 几乎相同
```

两个佐证:叶子读取在 CPU 和 CUDA 上耗时几乎一样(101 vs 106 ms),说明那段与设备无关、
GPU 全程干等;而核函数在 GPU 上快 29 倍 —— 正因为算得太快,读取的占比才被顶到 64%。

~~提速方向:预取 / 异步传输 / 在算第 N 块时后台读第 N+1 块~~(已作废,见上)。
**不要动**分块粒度和逐股票循环,两者都已被实测排除。

### 其他实测结论

- **分块粒度不是瓶颈**:`--chunk-symbols` 从自动(~15 只)放大到 64,速度只变 6%,显存多吃 10 GB。
- ~~成本单位是"读了几个字段",不是"跑了几棵树"~~ —— 修复前其实是"叶子被**用**了几次",随树数线性涨。
  修复后读取确实按字段计(每块每字段一次),但它只剩 5%;大头的核函数随**节点数**涨。
  population 64 的依据要按新数据重看。
- **单代耗时由缓存命中率决定**,与代数无单调关系,实测 8 分钟到 10 小时都出现过。
  **别用少数几代外推总时长** —— 这个错误犯过三次。
- 磁盘阵列顺序读 **3.5 GB/s**,任务峰值约 800 MB/s,有余量并行第二个任务(但 GPU 只有一张卡)。
- CPU 有 208 核,任务长期只用 1 核(100–115%),754 GB 内存通常只用到 30–80 GB。

---

## 10. 工作约定

```bash
cd ~/Desktop/AlphaCube && PYTHONPATH=src ~/.venvs/alphacube/bin/python -m pytest tests -q
```

当前 **324 passed, 8 skipped, 1 xfailed**。改完必跑。改完也要 `git push` 并 rsync 到服务器:

```bash
rsync -a --exclude __pycache__ --exclude .git --exclude .pytest_cache --exclude mining_runs \
  --exclude tmp --exclude '*.log' --exclude .DS_Store \
  -e "ssh -o BatchMode=yes -o ControlPath=none -p 29982" \
  ~/Desktop/AlphaCube/ root@connect.westd.seetacloud.com:/root/autodl-tmp/AlphaCube/
```

**注意:同步到服务器不影响正在跑的进程** —— Python 启动时已把代码载入内存,改动要下次启动才生效。

几个约定:

- 每个修复都要有**能复现该 bug 的测试**,注释写清楚它当初是怎么发生的
- 判据尽量**探针实测**,不手写名单
- CLI 参数有单独的测试(遍历 `cmd_mine` 读的每个 `a.xxx`,断言 parser 定义了它)——
  因为其他测试直接调 `run_mining` 不走 CLI,曾漏掉一个参数导致启动即失败
- **所有性能判断以实测为准。** 这个项目里已经猜错过三次(分块粒度、CPU 相对 GPU 的倍率、
  面板缓存失效),每次都是拿局部或简单负载外推整体
- 停掉正在跑的长任务前先问用户 —— 那可能是十几个小时的计算
