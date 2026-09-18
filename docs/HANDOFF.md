# AlphaCube 交接(2026-09-19 06:20 北京时间)

你正在接手 AlphaCube。这份文件自足 —— 读完就能接着干,不需要任何对话历史。
设计细节在 `docs/design.md`(中文,权威);这份讲**现状、地址、怎么跑、踩过什么坑、还欠什么**。
用户用中文交流,喜欢简洁直接的汇报;所有回测结论一律给**扣成本**口径。

---

## 0. 先读这里

**现在没有任务在跑,服务器数据盘卡死(§4;06:18 复查仍是 `write-pending`、校验仍停在 79.9%、gpu9 仍是 D 状态)。在它恢复之前,不要往 `/root/autodl-tmp` 写任何东西**
—— 包括 rsync、scp、启动任务;写入会挂在内核里,连你的命令一起卡住。读是正常的。

接手后按顺序做:

1. **看数据盘恢复没有**(只读):
   ```bash
   ssh -o BatchMode=yes -o ControlPath=none -p 29982 root@connect.westd.seetacloud.com 'cat /sys/block/md0/md/array_state; cat /proc/mdstat'
   ```
   `write-pending` = 还卡着;`clean` / `active` = 恢复了。修复是系统层的事(§4),由用户或 AutoDL 处理,**你不能动**。
2. **恢复后收尾 gpu9**:它已被用户要求停止,SIGINT 已发出、在等写入返回才生效。确认
   `ps -p 332647` 已退出、`ls /root/autodl-tmp/AlphaCube/tmp/` 为空、`screen -ls` 里没有 `gpu9`。
   若进程还在,再 `kill -INT 332647`。
3. **同步仓库到服务器**(§5.2)。服务器上的 `docs/` 和新脚本 `scripts/gp_full_market.sh`、
   `scripts/gp_report_loop.sh`、`scripts/eval_checkpoint_costs.py`、`scripts/merge_archives.py` 都还没过去。
4. **下一次运行前先问用户 §11 的三个决定**,再用 `scripts/gp_full_market.sh gpu10` 起跑(§5.3)。

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

当前 **341 passed, 8 skipped, 1 xfailed**(约 2 分钟)。改完必跑。

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

- 每代那一行:`best / gen_best / gen_mean`(|IC|)、`archive`、`valid`、`similar`(本代被 rank 封禁拒的数)、
  `exemplars`(rank 样本数)。滚块那一行带 `rolled N`。
- 块结算表:`[第i块 仅本块]` 与 `[第i块 累计1~i]`,列 `sharpe`(无成本)、`sharpe_net`(扣 15bp)、
  `sharpe_capped`(限换手 30%)、`turnover`。**看 `sharpe_net`。**
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
| gpu9 | 03:44~(05:41 卡住) | 封 12 个 + rank 0.95,逐元素算子出搜索 | 第 3 块开头被数据盘卡住,用户要求停止(§4) |

### 因子家族(修复后的 gpu4~gpu9 共 1314 条,详见汇总 CSV)

| 家族 | 最佳 \|IC\| | 扣 15bp 夏普(已测) | 备注 |
|---|---|---|---|
| 成交额水平(`day_max/day_close/log/pow` of Amt) | 0.031 | **+0.14 ~ +2.54** | 最好的是平滑版 `sub(0.5, abs_(ts_mean(day_max(log(Amt)), 10)))`:日换手 7%,扣费 2.54,2018~2025 每年为正;已封 |
| 日内自相关 `day_ar1_coefficient` | 0.028 | −4.9 ~ **+1.65** | `ts_max(…(High/Low), 30~50)` 长窗版日换手 3~5%、扣费 1.0~1.65,但 rank IC 仅 0.013~0.019、2025 年转负;High/Close/Ret/Volume 已封,**Low 未封** |
| Durbin-Watson(各价格序列) | **0.0333** | −2.0 ~ −1.7 | `day_durbin_watson(inv(Vwap))` 等;日换手 ~45%;只封了 Close 版,其他序列与之 rank 相关 < 0.95 能过 |
| 日内峰度 `nt_ts_kurt` / `day_kurt` | 0.032 | −18.6 ~ −9.6 | 日换手 81~89%;`day_kurt(Ret)` 族 IC 为正但 rank IC 为负(尾部撑起) |
| 收益-成交额协方差 `day_cov(Ret, Amt)` | 0.030 | +0.92 | 未封 |
| 日内最大分钟收益 `day_max(Ret)`(彩票效应) | 0.022 | 未测 | 已封 |

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
| (本提交) | 新增 `scripts/gp_full_market.sh`、`gp_report_loop.sh`、`eval_checkpoint_costs.py`、`merge_archives.py`;重写本文件 | |

---

## 9. 读结果时的陷阱(都踩过)

- **checkpoint 里的 `sharpe` / `annualized_return` 是无成本的**;扣成本要用 §5.7 补算,或看块结算表的 `sharpe_net`。
- **符号**:GP 用 `|IC|` 做适应度,方向由组合层事后对齐。`IC = −0.024、夏普 = −3.3` 是一条**强**因子
  (翻转后 +3.3)。可疑的是 |IC| 高而 |夏普| 接近 0。
- **IC 与 rank IC 符号相反** = 少数极端值撑起的 IC(`pow_2(day_kurt(...))` 就是),不能当真。
- **回撤 −99%**:负 IC 因子复利八年的算术必然,要看翻转后的。
- **`coverage` 与 `n_days` 一起看**:恒为 0 的面板处处有限照样过 coverage;`n_days` 才问当天有没有截面区分度。
- **换手**:无成本夏普 4.0 配 89% 日换手,扣 15bp 后是负的。**永远看扣成本那套。**
- **archive 里大量重复**:同一因子套单调变换(`sigmoid`、`pow_2`、`abs_(abs_(x))`、`ts_max(x, 1)`、
  `sub(0.5, x)`)各算一条;数条数不等于数方向。
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

### 待用户决定(都提过,用户说先继续跑,未动)

1. **换手纳入选择**:GP 适应度扣换手惩罚 / 池子只收低换手因子 / 组合后的信号平滑再交易。这是扣费后
   能否稳定赚钱的关键。
2. **适应度改用 rank IC**:防 `pow_2(峰度)` 这种尾部撑起的 Pearson IC(gpu9 第 2 块头名 IC +0.028、
   rank IC −0.015、夏普 −2.4)。
3. **archive 内部按 rank 相关去重**(例如 ≥ 0.99 视为同一因子):防每块头名被几十种单调改写占满
   archive 和选择压力(gpu9 第 1 块前 12 名全是 `day_max(Ret)`)。

### 技术遗留

| | 说明 |
|---|---|
| 数据盘 RAID 卡死 | §4,等用户 / AutoDL |
| CUDA 上分块不变性不成立 | CPU 上逐 bit 不变(测试锁定);CUDA 按张量形状选归约路径,块行数一变 `day_sum` 差 ULP 级;`day_durbin_watson` 的 `E[x²]−E[x]² < 1e-8` 判常数行由舍入噪声决定,会翻 NaN。修法(两遍法方差)要改 vendored 核,需用户拍板 |
| `sliding_{top,bottom}_{sum,mean}_masked_p5` | 沿 dim 0 滚动,跨股票边界,结果随分块变;strict xfail 钉住。修法待定 |
| DW 族换价格序列能过 rank 封禁 | 与 DW(Close) 的 rank 相关 < 0.95;要彻底拦需降阈值或按算子封 |
| 日频 `Vwap` 口径 / `Return` 的 39 处 NaN 差异 | 与生产日频 store 比对的遗留,未细查 |
| `factorgcl` 组合 | 需要 `torch_geometric`,服务器未装(超出授权) |
| GP 不能断点续跑 | 停了只剩 checkpoint,重跑从第 1 代 |

---

## 12. 性能事实(全部实测,别再重新推测)

- **叶子读取已不是瓶颈**:修复后 S 块 0.44 s(读 0.10 s)、D 块 0.46 s(读 0.05 s);大头是核函数,
  D 轮(NT 面必须逐日调核 + 边界文件跨步读)成了主要部分。预取 / 异步传输不值得做。
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
