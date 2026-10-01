"""多空书拆开看:两条腿各贡献多少,其中多少是交易不到的。

十分位多空书(多头前 10%、空头后 10%,各占一半仓位)拆成两条相对等权全市场的腿:
  多头腿 = 前 10% 等权收益 − 全市场等权收益
  空头腿 = 全市场等权收益 − 后 10% 等权收益
多空 = (多头腿 + 空头腿) / 2。纯多头相对全市场的超额,拿得到的只是多头腿;空头腿只能靠真做空。

每条腿再在几种"能不能交易"的口径下各算一遍,口径一层层收紧:
  原标签          标签怎么记就怎么算
  开盘能成交      多头剔除次日开盘封涨停(买不进),空头剔除次日开盘封跌停(融到券也卖不出去)
  空头只做流动性前 x%  空头腿只在近 20 日日均成交额排名前 x% 的票里选(融券标的的粗代理:
                  立方体里没有融券名单;小票基本借不到券)
入选都在"能交易的票"里重新排名,不是选完再扔。成本单边 15bp,按各条腿自己的换手收。

  cd /root/autodl-tmp/AlphaCube && PYTHONPATH=src python -u scripts/ls_anatomy.py \\
      --inputs _bench/tradable/market_20180102_20251231.npz \\
      --signal 原冠军=_bench/champion_signal.npy --signal 新冠军=_bench/champion_signal_buyable.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HALFLIVES = (0.0, 1, 2, 3, 5, 10, 20)


def daily_amount(store_path: str, start: str, end: str, chunk: int = 256) -> np.ndarray:
    """(D, S) 每天的成交额(元,不复权口径就是真金白银),停牌日 NaN。"""
    from alphacube.research import warmed_view
    from alphacube.store import open_cube_store

    view, _ = warmed_view(open_cube_store(store_path), start, end, warmup=60)
    out = np.full((view.n_days, view.n_symbols), np.nan)
    for a in range(0, view.n_symbols, chunk):
        b = min(a + chunk, view.n_symbols)
        amt = view.read_s("Amt", a, b).astype(np.float64)
        out[:, a:b] = np.where(np.isfinite(amt), amt, 0.0).sum(-1).T
    out[~view.tradable().T] = np.nan
    return out


def trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """沿天的过去 n 天均值(含当天,只用有限值;一个都没有就 NaN)。"""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.where(np.isfinite(x), 1.0, 0.0)
    cv, cc = np.cumsum(v, 0), np.cumsum(c, 0)
    sv, sc = cv.copy(), cc.copy()
    sv[n:] -= cv[:-n]
    sc[n:] -= cc[:-n]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sc > 0, sv / sc, np.nan)


def leg_weights(alpha: np.ndarray, eligible: np.ndarray, frac: float, top: bool) -> np.ndarray:
    """(D, S) 等权一条腿,权重和为 1。选法和 metrics.long_short_weights 逐行一致(稳定排序、
    k = max(1, floor(n*frac))、少于 2 只的日子空仓),只是只在 eligible 的票里排。"""
    a = np.where(eligible, np.asarray(alpha, dtype=np.float64), np.nan)
    fin = np.isfinite(a)
    n = fin.sum(axis=1)
    order = np.argsort(np.where(fin, a, np.inf), axis=1, kind="stable")      # 没值的排最后
    k = np.maximum(1, np.floor(n * frac).astype(np.int64))
    pos = np.arange(a.shape[1])[None, :]
    ok = (n >= 2)[:, None]
    if top:
        pick = ok & (pos >= (n - k)[:, None]) & (pos < n[:, None])
    else:
        pick = ok & (pos < k[:, None])
    w = np.zeros_like(a)
    np.put_along_axis(w, order, np.where(pick, 1.0 / k[:, None], 0.0), axis=1)
    return w


def leg_gross(w: np.ndarray, label: np.ndarray):
    """一条腿的 (日毛收益, 日换手):持仓里没有标签的票按 0 收益算(和 portfolio_returns 一致)。"""
    from alphacube.metrics import portfolio_returns

    _, gross, turn = portfolio_returns(w, label, 0.0)
    return gross, turn


def legs(alpha: np.ndarray, label: np.ndarray, long_ok: np.ndarray, short_ok: np.ndarray,
         cost: float = 0.0015, frac: float = 0.1):
    """逐日的 (多头腿, 空头腿, 多空, 空头腿相对本池):两条腿都相对等权全市场,各扣自己的成本;
    多空 = 两条腿之和的一半 —— 不加限制时和 long_short_weights 那本书逐日相等。
    空头池比全市场窄的时候(比如只做流动性好的),"空头腿"里会混进"池子本身相对全市场"的
    风格收益;"相对本池"那一列减的是空头池自己的等权收益,剩下的才是选股。"""
    fin = np.isfinite(label)
    r_mkt = np.where(fin, label, 0.0).sum(1) / np.maximum(fin.sum(1), 1)
    pool = fin & short_ok & np.isfinite(alpha)
    r_pool = np.where(pool, label, 0.0).sum(1) / np.maximum(pool.sum(1), 1)
    lpool = fin & long_ok & np.isfinite(alpha)
    r_lpool = np.where(lpool, label, 0.0).sum(1) / np.maximum(lpool.sum(1), 1)
    g_long, t_long = leg_gross(leg_weights(alpha, long_ok, frac, True), label)
    g_short, t_short = leg_gross(leg_weights(alpha, short_ok, frac, False), label)
    long_ex = g_long - r_mkt - 2.0 * cost * t_long          # 买进卖出各收一次
    short_ex = r_mkt - g_short - 2.0 * cost * t_short       # 做空赚的是它跌,成本照收
    short_pool = r_pool - g_short - 2.0 * cost * t_short
    long_pool = g_long - r_lpool - 2.0 * cost * t_long
    return long_ex, short_ex, 0.5 * (long_ex + short_ex), short_pool, long_pool


def sharpe(x: np.ndarray) -> float:
    f = x[np.isfinite(x)]
    return float(f.mean() / f.std() * np.sqrt(252)) if f.size > 2 and f.std() > 0 else float("nan")


def anatomy(alpha: np.ndarray, label: np.ndarray, up_lock: np.ndarray, down_lock: np.ndarray,
            liquid_rank: np.ndarray, days: slice, cost: float = 0.0015, frac: float = 0.1,
            liquid_cuts: Sequence[float] = (0.5, 0.3)) -> Dict[str, Dict[str, float]]:
    """返回 {口径: {long, short, ls 的年化与夏普}},都只在 days 上算。

    up_lock / down_lock 是"成交日"的封板(第 t 行 = 决策日 t-1 的成交日),这里自己对齐到决策日。
    liquid_rank:决策日的流动性分位(0 = 最活跃),NaN = 没法排。"""
    D = alpha.shape[0]
    a, lab = alpha[days], label[days]
    ent_up = np.zeros_like(up_lock)
    ent_dn = np.zeros_like(down_lock)
    ent_up[:-1], ent_dn[:-1] = up_lock[1:], down_lock[1:]          # 决策日 t 的成交日是 t+1
    ent_up, ent_dn, liq = ent_up[days], ent_dn[days], liquid_rank[days]
    every = np.ones_like(a, dtype=bool)
    out: Dict[str, Dict[str, float]] = {}

    def record(name, long_ok, short_ok):
        long_ex, short_ex, ls, short_pool, long_pool = legs(a, lab, long_ok, short_ok, cost, frac)
        out[name] = {"long_ann": float(np.nanmean(long_ex) * 252), "long_sh": sharpe(long_ex),
                     "lpool_ann": float(np.nanmean(long_pool) * 252), "lpool_sh": sharpe(long_pool),
                     "short_ann": float(np.nanmean(short_ex) * 252), "short_sh": sharpe(short_ex),
                     "pool_ann": float(np.nanmean(short_pool) * 252), "pool_sh": sharpe(short_pool),
                     "ls_ann": float(np.nanmean(ls) * 252), "ls_sh": sharpe(ls)}

    record("原标签", every, every)
    record("开盘能成交", ~ent_up, ~ent_dn)
    for cut in liquid_cuts:
        record(f"空头只做流动性前{cut:.0%}", ~ent_up, ~ent_dn & (liq <= cut))
    for cut in liquid_cuts:
        record(f"多头只做流动性前{cut:.0%}", ~ent_up & (liq <= cut), ~ent_dn)
        record(f"多头只做流动性后{1 - cut:.0%}", ~ent_up & (liq > cut), ~ent_dn)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="tradable_backtest.py prepare 的 npz")
    ap.add_argument("--signal", action="append", required=True, help="name=path.npy,可以给多个")
    ap.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    ap.add_argument("--amount", default="_bench/tradable/amount_day.npy", help="日成交额缓存,没有就现算")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.0015)
    a = ap.parse_args(argv)

    import tradable_backtest as tb
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows, long_short_weights, portfolio_returns

    t0 = time.time()
    mkt = tb.load_market(a.inputs, fill=False)              # 原标签口径:和多空 +3.98 那次一样
    label = forward_return(mkt.adj_label, 1, 1)
    D, S = label.shape
    split = max(1, min(D - 1, int(D * a.train_frac)))
    amt_path = Path(a.amount)
    if amt_path.exists():
        amount = np.load(amt_path)
    else:
        amount = daily_amount(a.store, mkt.dates[0], mkt.dates[-1])
        np.save(amt_path, amount)
    liq_val = trailing_mean(amount, 20)
    liquid_rank = np.full((D, S), np.nan)
    for t in range(D):
        f = np.isfinite(liq_val[t])
        n = int(f.sum())
        if n:
            o = np.argsort(-liq_val[t, f], kind="stable")
            r = np.empty(n)
            r[o] = (np.arange(n) + 0.5) / n
            liquid_rank[t, f] = r
    print(f"{mkt.dates[split]} 起是测试段;成交额就绪 ({time.time() - t0:.0f}s)", flush=True)

    for spec in a.signal:
        name, path = spec.split("=", 1)
        sig = np.load(path).astype(np.float64)
        # 半衰期按训练段多空净夏普选,和 build_champion 一致
        best, best_s = 0.0, -np.inf
        for hl in HALFLIVES:
            sm = ema_rows(sig, hl) if hl else sig
            net = portfolio_returns(long_short_weights(sm[:split], 0.1), label[:split], a.cost)[0]
            s = sharpe(net)
            if np.isfinite(s) and s > best_s:
                best, best_s = hl, s
        sm = ema_rows(sig, best) if best else sig
        print(f"\n=== {name}(hl{best:g},测试段;多空 = 两条腿之和的一半)===")
        print(f"{'口径':<16} | {'多头腿 年化':>10} {'夏普':>6} {'相对本池':>9} {'夏普':>6} | {'空头腿 年化':>10} {'夏普':>6} | "
              f"{'空头腿相对本池':>12} {'夏普':>6} | {'多空 年化':>9} {'夏普':>6}")
        test = slice(split, D)
        ent_up = np.zeros_like(mkt.up_lock); ent_dn = np.zeros_like(mkt.down_lock)
        ent_up[:-1], ent_dn[:-1] = mkt.up_lock[1:], mkt.down_lock[1:]
        le, se, ls_, _, _ = legs(sm[test], label[test], ~ent_up[test], ~ent_dn[test], a.cost)
        yrs = np.array([d[:4] for d in mkt.dates])[test]
        print("分年(开盘能成交口径,年内日收益累加):" + "  ".join(
            f"{y} 多头腿 {np.nansum(le[yrs == y]):+.1%} 空头腿 {np.nansum(se[yrs == y]):+.1%}"
            for y in sorted(set(yrs))))
        for k, v in anatomy(sm, label, mkt.up_lock, mkt.down_lock, liquid_rank, slice(split, D), a.cost).items():
            print(f"{k:<16} | {v['long_ann']:+10.1%} {v['long_sh']:+6.2f} {v['lpool_ann']:+9.1%} {v['lpool_sh']:+6.2f} | {v['short_ann']:+10.1%} "
                  f"{v['short_sh']:+6.2f} | {v['pool_ann']:+12.1%} {v['pool_sh']:+6.2f} | "
                  f"{v['ls_ann']:+9.1%} {v['ls_sh']:+6.2f}", flush=True)
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
