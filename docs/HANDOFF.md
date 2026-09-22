# AlphaCube 交接(2026-09-22 11:40 北京时间)

你正在接手 AlphaCube。这份文件自足 —— 读完就能接着干,不需要任何对话历史。
设计细节在 `docs/design.md`(中文,权威);这份讲**现状、地址、怎么跑、踩过什么坑、还欠什么**。
用户用中文交流,喜欢简洁直接的汇报;所有回测结论一律给**扣成本**口径。

---

## 0. 先读这里

**现在没有任务在跑,服务器数据盘卡死(§4;07:20 复查仍是 `write-pending`、校验仍停在 79.9%、gpu9 仍是 D 状态)。在它恢复之前,不要往 `/root/autodl-tmp` 写任何东西**
—— 包括 rsync、scp、启动任务;写入会挂在内核里,连你的命令一起卡住。读是正常的。
**容器里 `/sys` 是只读挂载**:停校验(`echo idle > sync_action`)在容器里做不到,连用户也不行,只能 AutoDL 在宿主机上处理。

**现状(09-22 11:40)**:磁盘已恢复(09-21 约 13:5x);**gpu12 已跑完 20 代**(结果见 §6);服务器空闲,GPU 空闲。
**gpu13 已备好、未启动**:`scripts/gp_full_market.sh gpu13` 已含 30 个封禁(新增 gpu12 四个块的 12 个)+ 5 个 archive 的
rank 样本(836 条,每批 150 条求值);代码含全部读盘修复(4da3e80 / 6c384f2 / 4d1594d)与 z 分数修复(108302e)。
**起跑前问用户**(用户 09-22 说"尝试修复到 13,但不要停止 gpu12";gpu12 此后已自然跑完)。起跑方法见 §5.3。

**挖出来的东西都在**:服务器 `mining_runs/*/checkpoints/` 完好;本地有全量备份(§2);全部表达式
去重汇总在 `~/Desktop/AlphaCube/mining_runs/全部因子表达式.csv`(1922 条)。

---

## 1. 这个项目是什么

三维立方体 `(股票, 交易日, 分钟)` 上的因子挖掘引擎:一个数据对象、一份算子注册表、一个引擎,
GP / 四个 RL miner / 因子池 / 组合 / 回测全跑在它上面。当前阶段是在真实全市场数据上跑 GP,
反复发现"搜索被钻空子"并堵漏(§10 每一条都是 GP 在真实数据上找到并利用过的),同时看扣成本后
能不能组合出可交易的信号(§6:能,但换手是主要瓶颈)。

---

## 2. 地址总表

### 本地(用户的 Mac)

| | |
|---|---|
| 仓库 | `~/Desktop/AlphaCube`(git,`main`,remote `https://github.com/xiangyu2024/AlphaCube.git`) |
| Python | `~/.venvs/alphacube/bin/python`(uv 建的;torch 2.14、numpy 2.4;有 MPS,无 CUDA) |
| 全部 run 的本地备份 | `~/Desktop/AlphaCube/mining_runs/`(git 已忽略):`gpu2`~`gpu9` 全部 `checkpoints/archive_genNNN.csv`,两个 `20260916_*` 早期 run 的 `members.csv` / `combination.csv` / `meta.json` |
| 运行日志备份 | `~/Desktop/AlphaCube/mining_runs/logs/gpu3_run.log` ~ `gpu9_run.log`;`gpu_run.log` 是 gpu2 的(8 代)。同级的 `replay/` 是空目录,可删 |
| 扣成本回测结果 | `~/Desktop/AlphaCube/mining_runs/evals/`:`top_gen003.json`(gpu4 第 3 代)、`top_gpu5_gen010.json`、`top_gpu7_gen005.json`、`top_gpu8_gen005.json`,以及 gpu6 不设上限重放第 1 块的 `block1_nocap.csv` / `block1_nocap_members.csv` |
| 全部表达式汇总 | `~/Desktop/AlphaCube/mining_runs/全部因子表达式.csv`(由 `scripts/merge_archives.py` 生成) |

### 服务器(AutoDL 容器)

| | |
|---|---|
| 连接 | `ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com`(公钥免密) |
| **必须带 `-o ControlPath=none`** | 否则会复用一个早已失效的 ControlMaster 套接字然后被拒 |
| 项目 | `/root/autodl-tmp/AlphaCube` |
| Python | `/root/miniconda3/envs/quant/bin/python`(**服务器上没有 `python3` 命令**) |
| 数据 | `/root/autodl-tmp/qmt_daily_data/fast_factor_compute`:分段 store,`minute/` 为基础(5207 只 × 2092 天),`incremental/` 下是追加段;8 个字段 Open High Low Close Volume Vwap Ret Amt,每个 10.5 GB,共 ~80 GB;全市场视图 20180102~20251231 = **1942 天 × 5220 只 × 241 分钟** |
| 挖掘产物 | `mining_runs/<run-id>/checkpoints/archive_genNNN.csv`(每代一份,archive 累积) |
| 运行日志 | `<run-id>_run.log`、`<run-id>_report.log`(在项目根目录) |
| 启动脚本 | 仓库里的 `scripts/gp_full_market.sh <run-id>` + `scripts/gp_report_loop.sh <run-id>`(§5.3);历史上用的 `scripts/gpu4.sh` ~ `gpu9.sh`、`gpu*_report.sh` **只在服务器上,不在 git 里** |
| 引擎临时目录 | `<项目>/tmp`(落盘边界文件,一个 9.8 GB;**别改回系统临时目录**) |
| `_bench/` | 本次会话的测量脚本与输出(A/B、逐算子一致性、分块不变性、池子基准、重放),可删;有用的已整理进 `scripts/` |
| 硬件 | GPU RTX 5090 32 GB;208 核;**容器内存上限 90 GiB**;系统盘 `/` 30 GB(已用 88%);数据盘 `/dev/md0` 3 TB RAID5(已用 66%) |

---

## 3. 硬约束

1. **服务器上只能改 `/root/autodl-tmp/AlphaCube`**,其他一切只读。装 pip 包、改系统设置(包括 RAID 的
   `sync_action`)都超出授权。
2. **系统盘只剩 ~4 GB**。中间产物一律落数据盘;引擎 `workdir` 默认是 `<项目>/tmp`。曾经写满系统盘、
   进程被静默杀掉、日志里一行报错都没有。
3. **内存按 90 GiB 算,不是 `free` 显示的 754 GB**(那是宿主机)。看容器真实占用用
   `/sys/fs/cgroup/memory.stat` 的 `anon`;`ps` 的 RSS 含可回收的 mmap 数据页,RSS 72 GB 时 anon 只有 20 GB。
   **池子不设上限会撑爆**(gpu6 就是这么静默死掉的,§6)。
4. **GPU 只有一张 32 GB**。在任务旁边跑任何 CUDA 程序,必须 `torch.cuda.set_per_process_memory_fraction(...)`
   加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 给自己封顶 —— 任务里 CUDA OOM 会被当成
   单个节点失败**静默吞掉**,不会崩,只会让那一代的树变成无效。任务启动时求样本那一步显存会冲到 31.7 GB。
5. **停掉正在跑的长任务前先问用户。** 用户决策很快;需要重启时,先把代码、测试、脚本都准备好,再问一次。

---

## 4. 当前事故:数据盘 RAID 卡死(2026-09-19 约 05:41 起)

- `/dev/md0`(3 块 NVMe 的 RAID5,挂在 `/root/autodl-tmp`)在做例行一致性校验(scrub),
  **卡在 79.9% 不动**(`/sys/block/md0/md/sync_completed` 一直是 11993399552 / 15002667008,
  标称速度 568 KB/s),`array_state` 停在 **`write-pending`**。3 块盘都在线、没降级(`[UUU]`)。
- 后果:**所有写入挂在内核 `md_write_start`**。gpu9 正在写落盘边界文件,进程停在 D 状态;
  容器 I/O 压力(`/sys/fs/cgroup/io.pressure`)97%。读不受影响。
- 能做的(**系统层,不在授权内,由用户或 AutoDL 客服决定**):停掉这次校验
  `echo idle > /sys/block/md0/md/sync_action`,或找 AutoDL 处理。
- 恢复后:按 §0 第 2、3 步收尾。gpu9 的 `tmp/alphacube_bvh5ou93/`(3 个边界文件)会在它退出时自动删除;
  若它是被强杀的,手动删这个目录。

---

## 5. 运行方法

### 5.1 本地测试

```bash
cd ~/Desktop/AlphaCube && PYTHONPATH=src ~/.venvs/alphacube/bin/python -m pytest tests -q
```

当前 **354 passed, 8 skipped, 1 xfailed**(约 2 分钟)。改完必跑。

### 5.2 同步到服务器

```bash
rsync -a --exclude __pycache__ --exclude .git --exclude .pytest_cache --exclude mining_runs \
  --exclude tmp --exclude '*.log' --exclude .DS_Store \
  -e "ssh -o BatchMode=yes -o ControlPath=none -p 29982" \
  ~/Desktop/AlphaCube/ root@connect.westd.seetacloud.com:/root/autodl-tmp/AlphaCube/
```

- 同步**不影响正在跑的进程**(模块已载入内存)—— 例外:`alphacube.combine.post_combine` 是**懒导入**的,
  第一次块结算时才加载,所以第 5 代之前同步的修复会被当前进程用上(gpu7 就这样用上了 NaN 修复)。
- 改完:`git push` + 上面的 rsync。数据盘卡着时只能 push,rsync 等恢复。

### 5.3 启动一次挖掘

```bash
ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com \
  'cd /root/autodl-tmp/AlphaCube && \
   screen -dmS gpu10 bash -c "./scripts/gp_full_market.sh gpu10 > gpu10_run.log 2>&1" && \
   screen -dmS gpu10rep bash -c "./scripts/gp_report_loop.sh gpu10"'
```

- `scripts/gp_full_market.sh <run-id>`:当前配置的样板,run-id 已存在会拒绝启动(不会覆盖旧产物)。
  改参数就改这个文件(提交进 git)。**长任务一律用 `screen`**(与 ssh 无关;`nohup` 试过不可靠)。
- 它的配置:20 代 = 4 块 × 5 代;池子上限 30(块结算取前 120 条、最终入池取前 60 条 —— 内存有界);
  单边 15bp、限 30% 日换手;**开局封禁 18 个结构**(§7)+ **rank 相关封禁 0.95**,样本取自
  gpu5 第 10 代、gpu7 第 7 代、gpu8 第 8 代、gpu9 第 10 代的 archive。
  gpu10 起另加:**适应度 `|rank IC| − 0.14 × 日换手`**、**因子族去重 0.99**、**组合后 EMA 平滑**
  (半衰期 1,2,3,5,10,20 在训练段选)—— 见 §11 与 `docs/design.md` §7。
  gpu10 的日志行多一个 `dup`(本代被判为同族而降级的个体数);`valid` 是降级后的数,会比以前低;
  `best / gen_best / gen_mean` 现在是新适应度(可为负),不再是 |IC|。
- **每次重启都该把上一次运行块结束时新封的结构加进 `--ban`,并把它最新的 archive 加进 `--ban-archive`**
  (日志里 `封禁 3 个一层结构,累计生效 N 个: ...` 那一行就是键),这是用户定下的做法。
  键写错会在启动时报错(`check_ban_key`),不会静默不生效。
- 启动时间线(正常):载入并求值 rank 样本 16~22 分钟 → 第 1 代(初始种群 + 第一批子代)约 10~15 分钟
  → 之后每代 2~12 分钟 → 每块结束:结算 2~7 分钟 + 重建种群求值 15~25 分钟。整个 run 约 4~6 小时。

### 5.4 监控

```bash
S='ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com'
$S 'cd /root/autodl-tmp/AlphaCube && grep -E "^ +[0-9]+/20" gpu10_run.log | tail -3'   # 每代一行
$S 'cd /root/autodl-tmp/AlphaCube && tail -5 gpu10_run.log'                             # 代内进度(每 60 秒)
$S 'cd /root/autodl-tmp/AlphaCube && tail -3 gpu10_report.log'                          # 每 5 分钟:RSS / anon / GPU / 盘 / md0
$S 'cd /root/autodl-tmp/AlphaCube && sed -n "/块结束/,\$p" gpu10_run.log | grep -v Warn'  # 块结算表
$S 'cat /sys/block/md0/md/array_state; df -h / /root/autodl-tmp'
```

- 每代那一行:`best / gen_best / gen_mean`(适应度;gpu9 及以前是 |IC|,gpu10 起是 |rank IC| − 0.14×换手)、
  `archive`(gpu10 起 = 因子族数)、`valid`、`similar`(本代被 rank 封禁拒的数)、`exemplars`(rank 样本数)、
  `dup`(gpu10 起,本代同族降级数)。滚块那一行带 `rolled N`。
- 块结算表:`[第i块 仅本块]` 与 `[第i块 累计1~i]`,列 `sharpe`(无成本)、`sharpe_net`(扣 15bp)、
  `sharpe_capped`(限换手 30%)、`turnover`,gpu10 起加 `halflife`(训练段选出的 EMA 半衰期)、
  `sharpe_smooth`(平滑后扣 15bp)、`turnover_smooth`。**看 `sharpe_net` 和 `sharpe_smooth`。**
- **卡住怎么判断**:代内进度行超过几分钟不更新 → 看 `ps -o stat= -p <pid>`、`cat /proc/<pid>/wchan`、
  `top -H`。R + 单核满 + GPU 闲 = Python 循环(见 §10 的逐元素算子);D + `md_write_start` = 磁盘(§4)。
  `py-spy` 用不了(容器禁 ptrace)。

### 5.5 从 Claude Code 等待服务器事件

- **`Monitor` 工具里 ssh 连不上服务器**(实测两次各 30 分钟零事件),用 `Bash` + `run_in_background`
  + `until`/`for` 循环轮询。
- 后台 Bash 单次上限 10 分钟:循环写成 ≤ 9 分钟(如 18 × 30 秒),到点再挂下一段。
- 用户的 Mac 休眠时本地轮询会暂停 —— 服务器上的 `*_report.log` 才是可靠的时间线。

### 5.6 停止

```bash
$S 'kill -INT <pid>; screen -S <run-id>rep -X quit'
```

SIGINT 走 KeyboardInterrupt → 引擎的 `finally` 删临时边界文件(SIGKILL 不会,9.8 GB 的文件会留在 `tmp/`)。
**GP 没有断点续跑**:archive 只在内存里,停掉后只剩 checkpoint;重跑从第 1 代开始。同配置同 seed
**逐字节可复现**(gpu4/5/6 的前几代 archive 完全相同),所以重跑前几代是重复计算、不是新结果。

### 5.7 给 checkpoint 头部因子补算扣成本回测

checkpoint 自带的指标是**无成本**的。在服务器上:

```bash
cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/quant/bin/python scripts/eval_checkpoint_costs.py \
  mining_runs/gpu9/checkpoints/archive_gen010.csv _bench/top_gpu9_gen010.json 20 0.045
```

第 4 个参数是 GPU 显存占比上限;**有任务在跑时用 0.045 或 0(0 = 只用 CPU,20 条约 2~20 分钟)**。
输出翻转到可交易方向后的无成本 / 扣 15bp / 限换手夏普、换手、逐年盈亏;先核对无成本数字与 checkpoint 一致。
算完把 JSON 拷回本地 `mining_runs/evals/`,汇总表就会带上它。

一个 run 的完整回测报告(单因子 + 全部组合方法的样本外净值、逐年逐月,JSON):

```bash
cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/quant/bin/python scripts/report_run.py \
  mining_runs/gpu12/checkpoints/archive_gen020.csv _bench/gpu12_report_cum.json --top-factors 40 --pool-top 120
```

GPU 空闲时 `--gpu-fraction 0.95`(默认),约 15 分钟;`--pool-top 60` 是最终入池的口径。

给一个 archive 补一次和块结算同口径的池 + 组合(含扣成本、限换手、组合后 EMA 平滑):

```bash
cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/quant/bin/python scripts/combine_archive.py \
  mining_runs/gpu9/checkpoints/archive_gen010.csv --top 120 --gpu-fraction 0.045 --out _bench/gpu9_gen010_combine
```

旁边有任务在跑时同样用 0.045 或 0;内存约 8~10 GB(120 个面板 + 30 个 z 面板),先看 anon 有没有余量。

### 5.8 汇总全部表达式

```bash
# 先把服务器上的 checkpoint 拉到本地(只读,不受 §4 影响)
rsync -a --include='*/' --include='checkpoints/***' --include='members.csv' --include='combination.csv' \
  --include='meta.json' --exclude='*' -e "ssh -o BatchMode=yes -o ControlPath=none -p 29982" \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/AlphaCube/mining_runs/ ~/Desktop/AlphaCube/mining_runs/
cd ~/Desktop/AlphaCube && PYTHONPATH=src ~/.venvs/alphacube/bin/python scripts/merge_archives.py mining_runs
```

每个 run 取最新 checkpoint,按表达式结构去重,标出处、核心结构、是否已封(脚本里的 `BANNED`,
**新封了结构要同步改**)、是否含已移出搜索的漏洞算子,并合入 `mining_runs/evals/*.json` 的扣成本结果。

### 5.9 CLI 参数速查(`python -m alphacube.cli mine --help`)

本次会话新增的:

| 参数 | 含义 |
|---|---|
| `--pool-capacity 0` | 池子不设上限(0 = 不限)。**会爆内存,别用**(§6 gpu6) |
| `--block-combine-top N` | 块结算取 archive 前 N 条(默认 120;0 = 全部) |
| `--ban KEY`(可重复) | 开局即封禁的一层结构,键如 `O:log(F:Amt)`、`O:nt_ts_argmax(F:High)\|w3`、`O:in_cov(F:Ret,F:Close)\|w30` |
| `--ban-rank-corr R` | 候选与任一被封家族样本的日截面排名相关 \|ρ\| ≥ R 就拒(0 = 关)。单调改写与反号恰为 1.0 |
| `--ban-archive CSV`(可重复) | 旧 run 的 checkpoint;其中含 `--ban` 键的表达式作为开局样本(需配 `--ban-rank-corr`) |
| `--fitness {ic,rank_ic}` | GP 适应度的底:\|日均 Pearson IC\| 或 \|日均 rank IC\|(默认 ic) |
| `--turnover-penalty λ` | 适应度再减 λ × 十分位多空书日换手(0 = 不扣;0.14 = 15bp 折成 rank IC 单位) |
| `--archive-dedup R` | 日排名相关 ≥ R 的树算同一因子族:archive 留最好的,同代只让它繁殖(0 = 关) |
| `--smooth-halflives 1,2,...` | 每个组合再按逐股 EMA 交易一遍,半衰期只在训练段按扣成本夏普选(空 = 关) |

---

## 6. 运行历史与结果(北京时间)

| run | 时间 | 代码 / 配置 | 停在哪、为什么 |
|---|---|---|---|
| gpu2, gpu3 | 09-17~18(上一任) | 漏洞修复前 | 结果含布尔洗白、float32 下溢等漏洞产物,\|IC\| 0.07~0.08 不可信 |
| gpu4 | 09-18 06:54~09:26 | 旧引擎,池 30 | 第 5 代中途,用户批准换新引擎 |
| gpu5 | 09:27~11:34 | 读取修复后 | 第 10 代后。第 1~4 代与 gpu4 **逐行相同**,耗时 137.6 → 37.5 分钟(3.7 倍) |
| gpu6 | 11:34~约 12:40 | 不设上限 | 第 1 块结算中途**静默死亡**(内存,无 traceback),前 5 代与 gpu5 逐字节相同 |
| gpu7 | 23:20~00:33 | 池 30 + 开局封 6 个 | 第 7 代后,换 rank 封禁 |
| gpu8 | 09-19 00:34~03:43 | 封 9 个 + rank 0.95 | 第 9 代**卡死**(`in_mask_nearby_ones`,ETA 77 小时) |
| gpu9 | 03:44~(05:41 卡住) | 封 12 个 + rank 0.95,逐元素算子出搜索 | 第 3 块开头被数据盘卡住,用户要求停止(§4);09-21 磁盘恢复时已被 SIGKILL |
| gpu10 | 09-21 14:05~14:28 | 新适应度 + 族 0.99 + 平滑,462 条 rank 样本 | 启动阶段 D 轮 3.2 天/分钟(~10 h):跨步读预读放大 ~350 倍(§12),停掉修 |
| gpu11 | 14:35~17:14 | 同上 + 跨步读 `MADV_RANDOM` | 第 4 代后停:发现边界文件的 advice 跨调用失效(6c384f2),D 轮又回到 95 天/分钟。4 代 checkpoint 保留 |
| gpu12 | 09-21 17:14~09-22 05:40 | 同上 + 6c384f2 | **跑完 20 代**(11 h;前 4 代与 gpu11 逐行相同)。第 3 块起重建种群慢(每次 ~4.7 h):分段 store 绕过了读盘 advice(4d1594d 已修)。结果见下 |

### 因子家族(修复后的 gpu4~gpu9 共 1314 条,详见汇总 CSV)

| 家族 | 最佳 \|IC\| | 扣 15bp 夏普(已测) | 备注 |
|---|---|---|---|
| 成交额水平(`day_max/day_close/log/pow` of Amt) | 0.031 | **+0.14 ~ +2.54** | 最好的是平滑版 `sub(0.5, abs_(ts_mean(day_max(log(Amt)), 10)))`:日换手 7%,扣费 2.54,2018~2025 每年为正;已封 |
| 日内自相关 `day_ar1_coefficient` | 0.028 | −4.9 ~ **+1.65** | `ts_max(…(High/Low), 30~50)` 长窗版日换手 3~5%、扣费 1.0~1.65,但 rank IC 仅 0.013~0.019、2025 年转负;High/Close/Ret/Volume 已封,**Low 未封** |
| Durbin-Watson(各价格序列) | **0.0333** | −2.0 ~ −1.7 | `day_durbin_watson(inv(Vwap))` 等;日换手 ~45%;只封了 Close 版,其他序列与之 rank 相关 < 0.95 能过 |
| 日内峰度 `nt_ts_kurt` / `day_kurt` | 0.032 | −18.6 ~ −9.6 | 日换手 81~89%;`day_kurt(Ret)` 族 IC 为正但 rank IC 为负(尾部撑起) |
| 收益-成交额协方差 `day_cov(Ret, Amt)` | 0.030 | +0.92 | 未封 |
| 日内最大分钟收益 `day_max(Ret)`(彩票效应) | 0.022 | 未测 | 已封 |

### gpu11 前 4 代(新适应度,样本内单因子,扣 15bp、按 IC 方向交易)

头部是 `day_cov(Amt, ·)` 的时间序列统计量:rank IC −0.035~−0.049、**日换手 1.9~3.9%、扣费夏普 +2.0~+3.0**
(第 4 代头名 `ts_stddev(day_cov(Amt, …))` +3.02);它们的 Pearson IC 只有 −0.005 左右,在旧 |IC| 适应度下排不上。
每代 18~21 个个体因同族被降级,4 代 archive 135 个族;rank 样本存下 230 个(462 条去重)。

### gpu12 结果(修复 z 分数后离线重算,`scripts/report_run.py`;报告页 https://claude.ai/artifact/NR2N1n5xrkEyNCqDwwxa21)

- **组合(样本外 = 后 30% 交易日,2023-08-08 起,扣 15bp)**:累计池(archive 前 120 条 → 池 30)ridge **+2.45**
  (无成本 +3.19,日换手 8.6%)、lstsq +2.42、icir +1.60、ic +1.36、equal +1.31;最终池(前 60 条)ridge +1.52、
  lstsq +1.47、icir +1.08。三种树模型日换手 40~56%,扣费 −2.9 ~ −4.8;平滑对线性方法基本中性,对树模型能减亏
  (最终池 gbdt/lightgbm 半衰期 20 天后 +1.0 左右)。
- **单因子(archive 前 40 族,方向按训练段 IC 定)**:扣费夏普中位数 全期 +1.30 / 训练 +1.62 / 测试 +0.80;
  37/40 测试段为正、12/40 测试段 > 1;日换手中位数 3.3%。头部:`ts_stddev(day_cov(Amt, Amt), 30)` 全期 +3.02
  测试 +2.25;`ts_quantile_q75(cross_cov(day_max(High), day_var(Ret), 15), 5)` 测试 +2.39(换手 11%)。
  2025 年是弱年:40 个里 13 个当年亏损。
- **运行时块结算表(09-21 看到的)低估了后半段**:含 `log(·)` 的因子有 −inf,旧 `zscore_rows` 让当天整个截面 z 分数
  变 NaN(108302e 已修),第 3 块单独那张表整个为空、累计 1~4 与最终池里也夹着"死成员"。上面是修复后的数。

### 融合回测(池 30,样本外 = 后 30% 交易日)

| run / 块 | 最好的扣 15bp 夏普 | 限换手 30% 最好 | 规律 |
|---|---|---|---|
| gpu5 第 1 块 | (当时组合层不扣成本) | — | lightgbm 无成本 4.59 但日换手 62% |
| gpu6 第 1 块(不设上限重放,145 条) | ridge −0.13 | 1.45 | 模型类全负;当时 equal/ic/icir/lstsq 因 NaN bug 全空(已修) |
| gpu7 第 1 块 | equal +0.24 | equal 1.25 | 模型类 −3.1~−5.5 |
| gpu8 第 1 块 | ic +0.28 | ic 1.25 | 同上 |
| gpu9 第 1 块 | equal −0.31 | equal 0.73 | 强家族封光后变弱 |
| **gpu9 第 2 块 / 累计 1~2** | **ic 加权 +1.50**(日换手 32~34%) | lstsq 1.68、ic 1.63 | **首次扣费后稳定为正**;树模型仍 −1.9~−5.5 |

**一贯的结论**:组合无成本夏普 4~5,但日换手 40~70%,15bp 吃掉绝大部分;能在扣费后为正的
都是换手最低的简单加权。**换手是当前的主要瓶颈**(§11)。

---

## 7. 已封禁的 18 个结构

下一次运行开局就该全部封掉(`scripts/gp_full_market.sh` 已含):

| 来自 | 结构 |
|---|---|
| gpu5 第 1 块 | `log(Amt)`、`nt_ts_argmax(High)\|w3`、`cross_decay_linear(Amt)\|w3` |
| gpu5 第 2 块 | `day_ar1_coefficient(High)`、`day_ar1_coefficient(Close)`、`day_durbin_watson(Close)` |
| gpu7 第 1 块 | `nt_ts_kurt(Vwap)\|w30`、`pow_1_5(Amt)`、`day_max(Amt)` |
| gpu8 第 1 块 | `day_ar1_coefficient(Volume)`、`day_ar1_coefficient(Ret)`、`in_cov(Ret,Close)\|w30` |
| gpu9 第 1 块 | `day_min(Ret)`、`day_max(Ret)`、`day_quantile_q25(Volume)` |
| gpu9 第 2 块 | `nt_ts_autocorr_s1(Ret)\|w3`、`pow_2(Ret)`、`day_kurt(Ret)` |

---

## 8. 本次会话的改动(按 commit,全部有能复现问题的测试)

| commit | 改了什么 | 效果 / 证据 |
|---|---|---|
| d275c24 | 分段 store 读取改切片(原来三轴 `np.ix_` 逐元素 gather);修重复行 bug | 生产块 58 ms → 2.1~2.7 ms |
| a5a00fc | 引擎一块内每个操作数只读一次(原来每用一次叶子读一次,一块 104 次读 8 个字段);探针锁定"核函数不写输入" | 同一批树 17.3 s → 5.6 s,输出逐 bit 相同;生产 S 轮 5.0 倍、D 轮 2.2 倍,一代 ~22 → ~7 分钟 |
| aab897d, 2872ef5 | CUDA 逐算子一致性跑通(240 个算子 0 缺陷);记录 CUDA 分块不变性不成立;生产形状实测 | 见 §12 |
| 52eba66 | `MsePool` 改增量 Gram 矩阵(原来每插一条在 ~800 万行上重解最小二乘,池满解两次) | 单次插入 55.5 → 5.1 s,成员与池 IC 不变 |
| 7f723f9 | 组合层接上成本与换手上限(原来块报告/最终表永远只有无成本夏普);`gbdt` 建模加种子(原来同输入两次夏普 5.0/3.0) | |
| 5ca8ecf | 滚块那行日志报真实数字(原来写死 `valid 0`) | |
| 1082bd0 | 支持不设上限:`--pool-capacity 0`、`--block-combine-top 0`、批量入池 `PoolBase.admit()`、向量化去重 | 每条候选 1.8 s;但**不设上限会爆内存,用户已决定设回上限** |
| 7e50681 | 组合层缺值按 0(中性)计,lstsq 不再要求所有成员同时有值 | 原来成员一多整张表 NaN、lstsq 静默退回等权 |
| 8f5ca07 | `--ban`:开局封禁结构,键有校验 | |
| 29e2559 | rank 相关封禁 `--ban-rank-corr` / `--ban-archive`(`mining/rankban.py`) | 单调改写(含反号)恰为 1.0,全部拦住 |
| a30d06f | 注册表构建时探针:数算子执行的 Python 行数随输入是否增长,逐元素循环的算子退出搜索(ND 面豁免) | 抓到 `in_mask_nearby_ones`(gpu8 卡死元凶) |
| 53f6273 | 新增 `scripts/gp_full_market.sh`、`gp_report_loop.sh`、`eval_checkpoint_costs.py`、`merge_archives.py`;重写本文件 | |
| (本提交) | §11 三个决定落地:`--fitness rank_ic --turnover-penalty 0.14`、`--archive-dedup 0.99`(`rankban.CompactRankIndex`)、`--smooth-halflives`(`post_combine.smoothed_metrics`、`metrics.ema_rows`);十分位书改稳定排序并向量化;`rank_rows` 不再给 inf 排名;checkpoint 新列;`merge_archives.py` / `combine_archive.py` 跟上 | 新适应度与报告的 rank IC、换手逐位一致(测试锁定);块末 rank 封禁按族成员判定;353 passed |

---

## 9. 读结果时的陷阱(都踩过)

- **checkpoint 里的 `sharpe` / `annualized_return` 是无成本的**(且按表达式原方向);扣成本要用 §5.7 补算,
  或看块结算表的 `sharpe_net`。gpu10 起 checkpoint 前 10 行有 `sharpe_net`(扣 15bp、按 IC 方向交易)和 `turnover`,
  每行都有 `score_ic` / `score_rank_ic` / `score_turnover`(适应度的原料)和 `dups`(族成员数);
  `fitness` 列不再是 |IC| —— `merge_archives.py` 已按新列处理。
- **符号**:GP 用 `|IC|` 做适应度,方向由组合层事后对齐。`IC = −0.024、夏普 = −3.3` 是一条**强**因子
  (翻转后 +3.3)。可疑的是 |IC| 高而 |夏普| 接近 0。
- **IC 与 rank IC 符号相反** = 少数极端值撑起的 IC(`pow_2(day_kurt(...))` 就是),不能当真。
- **回撤 −99%**:负 IC 因子复利八年的算术必然,要看翻转后的。
- **`coverage` 与 `n_days` 一起看**:恒为 0 的面板处处有限照样过 coverage;`n_days` 才问当天有没有截面区分度。
- **换手**:无成本夏普 4.0 配 89% 日换手,扣 15bp 后是负的。**永远看扣成本那套。**
- **archive 里大量重复**:同一因子套单调变换(`sigmoid`、`pow_2`、`abs_(abs_(x))`、`ts_max(x, 1)`、
  `sub(0.5, x)`)各算一条;数条数不等于数方向。
- **一个 inf 会抹掉一整天的 z 分数(已修,108302e)**:之前的 `zscore_rows` 用 nanmean/nanstd,把 inf 当值;
  含 `log(·)` 的因子常有 −inf,于是当天整个截面 NaN,池的闸却都屏蔽 inf 照样放行 —— 108302e 之前的池/组合结果里,
  这类因子是"死成员",数字偏低。
- **全部是样本内**:GP 用 2018~2025 整段的 IC 选因子,组合层的"样本外"只是后 30% 交易日;
  多空口径没算 A 股融券限制、涨跌停、冲击成本。
- **gpu2/gpu3 的高 IC 是漏洞产物**(汇总 CSV 里"含已移出搜索的算子"一列有值的都是)。

---

## 10. 已经堵上的漏洞(别再打开)

| 漏洞 | 现象 | 现在怎么防 |
|---|---|---|
| 平凡布尔算子 | `pow_05(eq(a,b))` 洗成合法根,\|IC\| 0.58 | `trivial_boolean` 探针,默认移出词表 |
| float32 下溢 | `pow(0.5, 成交量)` 面板恒零,\|IC\| 0.0741 | 二元 `pow` 移出搜索,指数烘焙为 `pow_05/1_5/2/2_5/3` |
| 部分退化面板 | 每天截面无方差,IC 来自少数天 | `--min-scored-days 0.5` |
| NT 面配对核 | `nt_ts_pearson` 一代 3~10 小时 | 按面排除(ND 的 `ts_pearson`/`ts_spearman` 保留) |
| 单结构霸榜 | `day_ar1_coefficient(Volume)` 占 archive 49% | 滚动块 + `--ban-top-n` |
| **单调改写绕过结构封禁** | 封了 `log(Amt)`,GP 写成 `log(pow_1_5(Amt))`,排名逐位相同 | `--ban-rank-corr 0.95`(用户定的口径:rank 相关) |
| **逐元素 Python 循环的算子** | `in_mask_nearby_ones` 一块 7 分钟,gpu8 第 9 代 ETA 77 小时 | 注册表构建时的行数探针,自动归入 resource_risk |
| **池子不设上限** | gpu6 在第 1 块结算静默死亡 | 用户决定设回上限 30 |
| **组合层 NaN 传染 / 不扣成本 / gbdt 无种子** | 表全空、只看无成本、结果不可复现 | 见 §8 |

> 判据一律用**探针实测**而非手写名单(`catalog._probe_constant_capable`、`_probe_per_element_python`)。

还修过的静默 bug:`pipeline.py` 里 `realized` 被重绑成截断版,最终 `compare()` 全部越界却被做成表格打印
—— 现在有测试断言最终表必须有数字。

---

## 11. 还没解决的

### 已由用户决定并实现(2026-09-19 07:00,gpu10 起生效)

1. **换手纳入选择**:用户选了"适应度扣换手" + "组合后平滑再交易"(没选"入池设换手门槛")。
   适应度 = `|rank IC| − 0.14 × 日换手`;组合层每种方法再试 EMA 平滑,半衰期只在训练段选。
2. **适应度改用 rank IC**:是。
3. **archive 内部按 rank 相关去重**:是,0.99;同代同族只让最好的繁殖(精确复制也算同族)。
4. 用户要求:磁盘恢复后**直接起跑 gpu10**,确认第 1 代正常后汇报。

依据与细节见 `docs/design.md` §7。**服务器实测(纯内存基准,07:40)**:新适应度每条新候选一次整面板
排序,1942×5220 在默认 104 个 torch 线程下 1.75 s/条,线程数按容器配额(`cpu.max` = 25 核)取 24 时
0.43 s/条 —— 所以打分时临时把 torch 线程设为配额(`gp.cpu_quota` / `torch_threads`),完后恢复;
族索引 64 候选 × 800 族一次 0.21 s。**还要验证的**:(a) 起跑后看每代耗时;(b) 族索引每族 1.5 MB,
内存随族数增长;(c) 块结算多了平滑,每块估计多 1~2 分钟。
**顺带发现、未动**:整个任务的 torch 都是 104 线程跑在 25 核配额上(超订 4 倍);引擎 CPU 侧如果有
torch 并行运算也在被节流。要不要全局设成配额,需要新旧 A/B(且 CPU 归约结果会随线程数变最后几位)。

### 技术遗留

| | 说明 |
|---|---|
| 数据盘 RAID 卡死 | 已恢复(2026-09-21 约 13:5x `array_state` 回到 `active`,校验从 79.9% 继续推进);§4 留作记录 |
| 校验还在跑 | 09-21 14:00 时 91.1%、21.9 MB/s(上限 `sync_speed_max` 20000 KB/s),约 8 小时后结束;读带宽实测未明显受影响 |
| CUDA 上分块不变性不成立 | CPU 上逐 bit 不变(测试锁定);CUDA 按张量形状选归约路径,块行数一变 `day_sum` 差 ULP 级;`day_durbin_watson` 的 `E[x²]−E[x]² < 1e-8` 判常数行由舍入噪声决定,会翻 NaN。修法(两遍法方差)要改 vendored 核,需用户拍板 |
| `sliding_{top,bottom}_{sum,mean}_masked_p5` | 沿 dim 0 滚动,跨股票边界,结果随分块变;strict xfail 钉住。修法待定 |
| DW 族换价格序列能过 rank 封禁 | 与 DW(Close) 的 rank 相关 < 0.95;要彻底拦需降阈值或按算子封 |
| 日频 `Vwap` 口径 / `Return` 的 39 处 NaN 差异 | 与生产日频 store 比对的遗留,未细查 |
| `factorgcl` 组合 | 需要 `torch_geometric`,服务器未装(超出授权) |
| GP 不能断点续跑 | 停了只剩 checkpoint,重跑从第 1 代 |

---

## 12. 性能事实(全部实测,别再重新推测)

- **跨步读的内核预读是 D 轮的真正瓶颈(2026-09-21 实测并已修)**:立方体是 `(S, D, M)`,D 轮"多股票 ×
  少数天"的读法每 1.9 MB 跨步里只要 ~1 KB,内核却围着每个碎片预读一整片。生产 store 上(9.8 GB 字段、
  全部 5207 只 × 6 天、每次先 `POSIX_FADV_DONTNEED` 清缓存,连测两轮数字一致):
  **旧 9.04 s / 读 10014 MB(要 28.7 MB,放大 349 倍)→ 新 1.05 s / 读 49 MB(1.7 倍)= 快 8.6 倍、少读 204 倍**;
  折算成整个字段 1942 天:49 分钟 → 5.7 分钟。修法是按访问模式下 `madvise`(`store.advise_access`):
  跨步读用 `MADV_RANDOM`,连续读保留预读(S 轮实测不受影响 0.06 → 0.05 s;若也关掉预读则只有 47 MB/s)。
  引擎的两个边界文件也是 `(S, D, M)`,D 轮里同样跨步读写,一并处理。
  **这就是 gpu10 启动阶段 D 轮只有 3.2 天/分钟(全程要 ~10 小时)的原因**;以前几次 run 没暴露,是因为
  它们首尾相接、页缓存是热的。生产 store 没有 D-major 镜像(`d_major_files` 不存在),所以这条路必然跨步。
- **叶子读取已不是瓶颈**:修复后 S 块 0.44 s(读 0.10 s)、D 块 0.46 s(读 0.05 s);大头是核函数。
  预取 / 异步传输不值得做。
- **一代 2~12 分钟**(视缓存命中),滚块后的新种群 15~25 分钟;**别用少数几代外推总时长**(犯过三次)。
- **块结算**(池 30、候选 120):约 2~7 分钟(修复前 46 分钟)。
- **内存**(面板缓存 1024 + 池 30):anon 20~45 GB;rank 样本每个约 31 MB。
- **显存**:任务常驻约 28.9 GB;启动时求 200~300 条样本冲到 30~31.7 GB。
- **分块粒度不是瓶颈**:`--chunk-symbols` 从自动(~15)放大到 64,速度只变 6%,显存多 10 GB。
- **CPU 上核函数比 GPU 慢约 29 倍**;208 核,任务长期只用 1 核。
- **数据盘**顺序读 3.5 GB/s;整个 store(80 GB)放不进 90 GiB 容器的页缓存,会被反复重读。

---

## 13. 工作约定

- 每个修复都要有**能复现该 bug 的测试**(先在旧代码上确认它会失败),注释写清楚当初怎么发生的。
- 判据尽量**探针实测**,不手写名单。
- CLI 参数有单独测试(遍历 `cmd_mine` 读的每个 `a.xxx`,断言 parser 定义了它)。
- **所有性能判断以实测为准**;要在服务器上测,就在 `_bench/` 或 `scripts/` 里写脚本,新旧代码 A/B,并核对输出逐 bit 相同。
- 改完:测试全绿 → `git commit`(结尾 `Co-Authored-By` 行)→ `git push` → rsync 到服务器。
- **停掉正在跑的长任务前先问用户**;重启要换新 run-id,别覆盖旧的 `mining_runs/<id>`。
- 汇报用中文,简洁,先说结论;回测数字给扣成本口径,并注明样本内 / 样本外。
