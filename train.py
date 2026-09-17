"""一步启动:读立方体 store → 挖因子 → 出指标 → 组合排行榜 → 全部落盘。

    python train.py

**所有参数都在文件末尾的 `run_mining(...)` 调用里显式写出**,没有一个靠默认值。
要改什么直接改那里。五个 miner 与 PPO 热启动、三种池、八种 label 都从这里选。
长任务请用 `python -u train.py > run.log 2>&1 &`。

数据来自 `ALPHACUBE_STORE`(默认服务器上的 V2 分钟 store 计算根目录,本机可指向
`python -m alphacube.cli synth` 生成的目录)。本脚本只读它。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacube.pipeline import run_mining  # noqa: E402

if __name__ == "__main__":
    run_mining(
        # ---------------- 数据 ----------------
        # 立方体 store:计算根目录(含 minute/ 与 incremental/)或单个分钟 store 目录
        store=os.environ.get("ALPHACUBE_STORE", "/root/autodl-tmp/qmt_daily_data/fast_factor_compute"),
        start="20200101",            # YYYYMMDD,自动吸附到交易日;None = 从头
        end="20231231",
        max_symbols=300,             # 只取 store 顺序里的前 N 只;None = 全市场
        symbols=None,                # 指定代码列表,给了就忽略 max_symbols
        warmup=60,                   # 在 start 之前多读的交易日数(跨日滚动窗口的预热)
        # ---------------- 预测目标 ----------------
        label="open_vwap15",         # open_vwap15 / close / volatility / downside_volatility / rank / class / spike / crash
        label_params={},             # 转发给 models.targets 的额外参数,如 {"mar": 0.0}
        horizon=1,
        # ---------------- 搜索 ----------------
        miner="gp",                  # gp / alphagen / alphaqcm / alphaforge / alphasage / warmstart
        budget=480,                  # GP:候选表达式条数(代数 = budget / population);RL:episode 数
        seed=0,
        device="cpu",                # "cuda" 上服务器;跨设备不承诺逐 bit
        chunk_symbols=None,          # None = 按 2 GB 预算自动定块
        chunk_days=None,
        gp_params=dict(population_size=24, max_depth=4, min_depth=2, top_k=None,
                       # 一张面板 = n_days x n_symbols x 4 字节(全市场 1942x5219 约 40 MB)。
                       # 缓存未命中要把这棵树碰到的每个字段重读一遍(每个 9.1 GB),
                       # 所以内存大的机器上这是最便宜的提速手段。512 约 21 GB。
                       panel_cache_items=512),
        rl_params=dict(),            # 例:dict(episodes_per_update=8) / dict(tail_alpha=0.9)
        max_expr_len=20,             # RL 每条表达式的 token 上限
        allow_resource_risk=False,   # 是否允许 NT 面的配对核(nt_ts_pearson / nt_ts_spearman)与 in_multireg。
                                     # 配对核都是 O(窗口数 x d x N^2)——全市场每窗口 2720 万对——但代价差在"面":
                                     # ND 面的 ts_pearson / ts_spearman 整个面板算一次,默认保留;
                                     # NT 面的双胞胎要逐日算 1942 次,约 241 倍,默认排除
        allow_boolean_ops=False,     # 是否让搜索使用 29 个严格 0/1 的算子(eq / is_gt0 / cs_quantile_q50 …)。
                                     # 默认否:它们本就不能当根,但 pow_05(eq(a, b)) 这类算术会把它们洗成合法根,
                                     # 得到的近似常数指示变量会拿到虚高的 IC
        # ---------------- 因子池 ----------------
        pool="mse",                  # mse / meanstd / dynamic
        pool_capacity=10,
        dedup=0.99,                  # 与任一成员 |corr| 超过此值即拒绝;None 关闭
        min_coverage=0.5,            # 候选在可交易股票日上的有限值占比下限
        min_scored_days=0.5,         # 候选必须在多大比例的可用交易日上真的算得出 IC。
                                     # min_coverage 只问"格点是不是有限值",一个恒为 0 的面板照样通过;
                                     # 这一条问的是"当天到底有没有截面区分度"            # 候选在可交易股票日上的有限值占比下限
        meanstd_lr=0.05, meanstd_steps=200, dynamic_window=60,
        seed_library=None,           # 例:PROJECT_ROOT / "configs" / "seed_alpha_library_dt.yaml"
        presqueeze=None,             # 例:dict(n_iterations=4, top_k_per_iteration=8)
        # ---------------- 组合 ----------------
        combine_methods=("equal", "ic", "icir", "lstsq", "ridge", "gbdt"),
        combine_train_frac=0.7,
        # ---------------- 输出 ----------------
        workdir=PROJECT_ROOT / "tmp",   # 落盘中间量:一个边界文件就是整块 (S, D, M) float32
                                       # (5219x1942x241 = 9.8 GB),绝不能落在小的系统盘上
        engine_progress_every=60,    # 每 N 秒打印一行"代内进度"(轮次/分块/百分比/ETA)。
                                     # 一代就是一次引擎调用,不开这个的话日志会静默整整一代
        checkpoint_every=1,          # 每 N 代把排好序的 archive 写进 <run>/checkpoints/。
                                     # 不设的话表达式只存在于内存里,跑到最后才落盘,中途被打断就全丢
        out_root=PROJECT_ROOT / "mining_runs",
        run_id=None,
        save_values=True,
    )
