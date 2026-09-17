# AlphaCube 交接说明

给接手这个项目的下一个 agent。读完这份就能接着干,不用从头翻对话。
设计层面的东西在 `docs/design.md`(中文,权威);这份只讲**当前在做什么**和**怎么操作**。

---

## 1. 一句话

三维立方体 `(股票, 交易日, 分钟)` 上的因子挖掘引擎。一个数据对象、一份算子注册表、
一个引擎,GP / 四个 RL miner / 池 / 组合 / 回测全跑在它上面。

**当前阶段**:在真实全市场数据上跑 GP,反复发现"搜索被钻空子"并堵漏。

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

**连接务必带 `-o ControlPath=none`** —— 否则会去复用一个早已失效的 ControlMaster 套接字:

```bash
ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com '<cmd>'
```

### 两条硬约束

1. **只能改 `/root/autodl-tmp/AlphaCube`**,服务器上其他一切只读。
2. **系统盘 `/` 只有 30 GB 且已用 86%**(miniconda 22 GB + vscode-server 8.4 GB)。
   任何中间产物都必须落在数据盘。引擎的 `workdir` 默认已是 `<项目>/tmp`,别再改回系统临时目录 ——
   一个落盘边界文件是整块 `(S, D, M)` float32 = **9.8 GB**,曾经把系统盘写满、进程被静默杀掉。

---

## 3. 跑一次挖掘

```bash
ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com \
  'cd /root/autodl-tmp/AlphaCube && screen -dmS gpu5 bash -c "./scripts/gpu4.sh > gpu5_run.log 2>&1"'
```

`scripts/gpu4.sh` 是当前配置的样板,直接改参数复用。**长任务一律用 `screen`**,
它与 ssh 连接无关,断网不影响。

### 当前配置的含义

```
--budget 1280 --population 64        20 代
--rolling-block 5 --ban-top-n 3      每 5 代推倒重来,封禁该块最常用的 3 个一层结构
--cost-per-side 0.0015               单边 15bp,买卖都收
--turnover-cap 0.30                  另出一套"日换手率 ≤30%"的曲线
--checkpoint-every 1                 每代把表达式+回测写盘
--progress-every 60                  代内进度(轮次/units/百分比/ETA)
--panel-cache 1024                   面板缓存条数(一张 40 MB,1024 ≈ 42 GB)
--min-scored-days 0.5                因子必须在过半可用交易日上真的算得出 IC
```

---

## 4. 监控

```bash
S='ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com'

$S 'cd /root/autodl-tmp/AlphaCube && tail -20 gpu4_run.log'          # 进度
$S 'cd /root/autodl-tmp/AlphaCube && ls mining_runs/gpu4/checkpoints' # 每代产物
$S 'ps -eo pid,etime,pcpu,rss,args | grep "[c]li mine"'               # 活没活着
$S 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'  # GPU
$S 'df -h / /root/autodl-tmp'                                          # 磁盘(系统盘必看)
```

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
中途杀掉进程,没有 checkpoint 就什么都拿不到 —— 这是花了一整夜才换来的教训,所以
`--checkpoint-every 1` 基本应该一直开着。

---

## 6. 读结果时的陷阱(都是踩过的)

**符号**:GP 用 `|IC|` 做适应度,方向由组合层事后对齐。所以 `IC = -0.024, 夏普 = -3.3`
是一条**强**因子(翻转后夏普 +3.3),不是坏因子。真正可疑的是 **|IC| 高而 |夏普| 接近 0**。
按 `|夏普|` 排序(`show2.py`)才对应真实强度。

**回撤 -99%**:负 IC 因子被复利八年的算术必然(年化 -44% 复利 7.7 年 = 净值剩 1%),
不是"某天暴雷"。看翻转后的。

**`coverage` 与 `n_days` 要一起看**:`coverage` 只问格点是否有限,一个恒为 0 的面板
处处有限、照样通过;`n_days` 问的是**当天有没有截面区分度**。两者背离就是退化因子。

**换手率**:现在每条因子都有 `turnover` / `*_net`(扣成本)/ `*_capped`(限换手)。
无成本的夏普 4.0 配 89% 日换手,扣 15bp 后是负的。**永远看扣成本那套。**

---

## 7. 已经堵上的漏洞(别再打开)

搜索会钻一切空子。以下每条都是在真实数据上被 GP 找到并利用过的:

| 漏洞 | 现象 | 现在怎么防 |
|---|---|---|
| **平凡布尔算子** | `pow_05(eq(a,b))` 把恒定指示量洗成合法根,\|IC\| 0.58 | `trivial_boolean` 探针(喂全正/全负/混号/同操作数,任一使输出恒定即判定),22 个默认移出词表 |
| **float32 下溢** | `pow(0.5, 成交量)`:0.5^1e5 = 0,面板恒零,\|IC\| 0.0741 全靠 329/1893 天 | 二元 `pow` 移出搜索,指数改为烘焙变体 `pow_05/1_5/2/2_5/3` |
| **部分退化面板** | 有值但每天截面无方差,IC 从少数天来 | `--min-scored-days 0.5` |
| **NT 面配对核** | `nt_ts_pearson` 逐日 5219² 对 × 1942 天,一代跑 3~10 小时 | 按**面**排除(ND 的 `ts_pearson` 保留,NT 双胞胎排除) |
| **单结构霸榜** | `day_ar1_coefficient(Volume)` 占 archive 的 49%,外面套不同单调变换 | `--rolling-block` + `--ban-top-n` |

> 判据一律用**探针实测**而非手写名单 —— 见 `catalog._probe_constant_capable`。

---

## 8. 还没解决的

| | 说明 |
|---|---|
| `sliding_{top,bottom}_{sum,mean}_masked_p5` | 声明为 `dt/in`(分钟轴)但 `_sliding_windows` 沿 **dim 0** 滚动,而引擎在该面喂进来的 `(A·B, M)` 里 dim 0 是**股票日** → 跨股票边界、结果随分块而变。已用 **strict xfail** 钉住(`test_sliding_masked_family_is_chunk_invariant`)。修法待定:改滚动轴 / 改注册到 ND 面 / 剔除 |
| 日频 `Vwap` 口径 | 新写法复现生产日频 store 到中位 8e-5,但不是原式,生产到底怎么算的未知 |
| `Return` 的 39 处 NaN 位置差异 | 与生产日频 store 比对时出现,约 0.8%,疑似复牌日,未细查 |
| CUDA 逐算子一致性 | `scripts/validate_gpu_parity.py --device cuda` 还没在服务器跑过(本机用 MPS 跑过,238 个算子零缺陷) |
| `factorgcl` 组合方法 | 需要 `torch_geometric`,服务器未装(装它要动 conda 环境,超出授权) |

---

## 9. 性能事实(实测,别再重新推测)

- **瓶颈不是分块粒度**。`--chunk-symbols` 从自动(~15 只)放大到 64,速度只变 6%,显存却多 10 GB。
  GPU 利用率长期 1–9%,时间花在每块内部的 Python 调度和逐股票/逐日循环上。
- **成本单位是"读了几个字段"**,不是"跑了几棵树"。单字段全量 9.1 GB,
  所以一代里多塞几棵树反而便宜 —— 这是 population 取 64 的依据。
- **单代耗时由缓存命中率决定**,与代数无单调关系,实测 8 分钟到 10 小时都出现过。
  别用少数几代外推总时长(这个错误犯过三次)。
- 磁盘阵列顺序读 **3.5 GB/s**,任务峰值约 800 MB/s,有余量并行第二个任务(但 GPU 显存只有一张卡的量)。

---

## 10. 测试

```bash
cd ~/Desktop/AlphaCube && PYTHONPATH=src ~/.venvs/alphacube/bin/python -m pytest tests -q
```

当前 **320 passed, 8 skipped, 1 xfailed**。改完必跑。

几个约定:
- 每个修复都要有**能复现该 bug 的测试**,注释写清楚它当初是怎么发生的
- 判据尽量**探针实测**,不手写名单
- CLI 参数有单独的测试(遍历 `cmd_mine` 读的每个 `a.xxx`,断言 parser 定义了它)——
  因为测试直接调 `run_mining`,不走 CLI,曾漏掉一个参数导致启动即失败
