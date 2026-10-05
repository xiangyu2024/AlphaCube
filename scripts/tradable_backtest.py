"""可交易的纯多头回测:真实资金、整百股、单票上限、停牌与涨跌停、按成交金额计费。

`long_only_grid.py` 交易的是"权重":每天把权重设成目标,换手按 sum|dw|/2 计,
资金无限可分。这里换成一个真账户:

  - 某一天放进 `capital` 元现金(默认测试段第一天放 100 万),之后复利滚动;
  - 主板/创业板持股数是 100 股的整数倍;科创板最少 200 股、之后按 1 股递增
    (按**不复权**价定股数,ΣAmt/ΣVolume);
  - 新建或加仓时单票金额 <= cap x 当日净值;
  - 决策日 t 收盘后出信号,t+1 开盘 15 分钟 VWAP 成交,持有到 t+2 的同一时点
    (和标签 open_vwap15 同一口径,收益用**后复权**价,分红送转都算在里面);
    科创板有 14~17% 的交易日分钟 Vwap 字段大面积缺失、标签算不出成交价 ——
    这里用不复权 VWAP15 x 当天的复权因子补齐(`fill_prices`),否则这些天它们既买卖不了
    也估不了值;
  - 停牌的票买不了也卖不了,按最后价格计市值;开盘 15 分钟封涨停的买不进,
    封跌停的卖不出;单笔成交不超过开盘 15 分钟成交量的 `participation`;
  - 费用按实际成交金额收(`Costs`:项目口径单边 15bp,或 A 股实际费率)。

组合构建有两种:
  greedy   用户给的目标 max sum_i w_i * exp(MinMax(signal_i)),s.t. 整百股、w_i <= cap。
           exp(MinMax(.)) 是信号的单调变换、而且恒 >= 1,所以这个线性目标的最优解
           就是"按信号降序,每只买到 cap,钱用完为止" —— exp/MinMax 只决定次序,
           **持几只完全由 cap 决定**(约 1/cap 只)。因为每只股票的得分都 >= 1,
           目标函数会把剩下的零钱塞进任何一只票,所以只在信号前 `pool_frac` 里分配。
  weights  给定目标权重(比如前 20% exp1),换算成股数再走同一套约束 —— 用来把
           `long_only_grid` 的 +1.92 在这套记账下复现出来,作为基线。

子命令:
  prepare  (服务器)从立方体算出成交价、不复权价、开盘 15 分钟量和高低价、收盘价,存 npz。
  run      读 npz 和信号,先复现基线,再扫 cap x 半衰期(在训练段选),报测试段。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alphacube.tradability import board_limit, ffill_rows, limit_locks   # noqa: E402  (测试也从这里拿)

LOT = 100
OPEN_WINDOW = (1, 16)           # 和标签同一个窗口:V2 网格的 09:31 .. 09:45


# ---------------------------------------------------------------------------
# 费用
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    """一笔成交的费用(元)。

    rate            按金额收的比例部分,每边都收(滑点、过户费;项目口径下就是全部成本)
    commission      佣金费率,和 min_commission 比大小,每笔单独算
    min_commission  每笔最低佣金(元)。小资金的真实摩擦主要在这里:5000 元的单子
                    万 2.5 只要 1.25 元,却要收 5 元,等于 10bp
    stamp           卖出印花税 ((起始日期, 税率), ...),按日期升序
    """
    name: str
    rate: float = 0.0015
    commission: float = 0.0
    min_commission: float = 0.0
    stamp: Tuple[Tuple[str, float], ...] = ()

    def stamp_rate(self, date: str) -> float:
        r = 0.0
        for start, v in self.stamp:
            if date >= start:
                r = v
        return r

    def one(self, notional: float, sell: bool, date: str) -> float:
        if notional <= 0:
            return 0.0
        fee = notional * self.rate
        if self.commission or self.min_commission:
            fee += max(notional * self.commission, self.min_commission)
        if sell:
            fee += notional * self.stamp_rate(date)
        return fee

    def many(self, notional: np.ndarray, sell: bool, date: str) -> np.ndarray:
        n = np.asarray(notional, dtype=np.float64)
        fee = n * self.rate
        if self.commission or self.min_commission:
            fee = fee + np.maximum(n * self.commission, self.min_commission)
        if sell:
            fee = fee + n * self.stamp_rate(date)
        return np.where(n > 0, fee, 0.0)

    def buy_rate(self) -> float:
        """不算最低佣金时的买入费率,估算"这些钱能买几手"用。"""
        return self.rate + self.commission


FLAT15 = Costs("单边15bp")
# 佣金万 2.5(最低 5 元)+ 过户费十万分之一 + 滑点 5bp;印花税 2023-08-28 起减半到万 5。
RETAIL = Costs("A股费率+5bp滑点", rate=0.0005 + 0.00001, commission=0.00025, min_commission=5.0,
               stamp=(("00000000", 0.001), ("20230828", 0.0005)))
COST_MODELS = {"flat15": FLAT15, "retail": RETAIL}


# ---------------------------------------------------------------------------
# 行情
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """(D, S) 面板,行 = 交易日,列 = 股票。第 e 行是"在第 e 天开盘 15 分钟成交"用的数。"""
    dates: List[str]
    adj: np.ndarray            # 后复权开盘 15 分钟 VWAP(标签同口径),计收益用
    raw: np.ndarray            # 不复权开盘 15 分钟 VWAP = ΣAmt/ΣVolume,定手数用
    vol15: np.ndarray          # 开盘 15 分钟成交股数
    up_lock: np.ndarray        # bool:开盘 15 分钟封涨停,买不进
    down_lock: np.ndarray      # bool:开盘 15 分钟封跌停,卖不出
    symbols: List[str] = field(default_factory=list)
    list_day: Optional[np.ndarray] = None   # (S,) 上市日在面板里的行号;窗口开始前就上市的是很小的负数
    lot_min: Optional[np.ndarray] = None    # (S,) 每笔最少股数(科创板 200,其余 100)
    lot_step: Optional[np.ndarray] = None   # (S,) 超过最少股数之后的递增单位(科创板 1,其余 100)
    adj_label: Optional[np.ndarray] = None  # 补齐之前的成交价(= 标签用的价格),复现基线用


def lot_rules(symbols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(每笔最少股数, 递增单位)。主板、创业板 100 股一手;科创板单笔不少于 200 股、
    超出部分按 1 股递增;北交所不少于 100 股、按 1 股递增。"""
    star = np.array([s.startswith(("688", "689")) for s in symbols], dtype=bool)
    bse = np.array([s.startswith(("4", "8", "92")) for s in symbols], dtype=bool)
    return np.where(star, 200.0, 100.0), np.where(star | bse, 1.0, 100.0)


def qty_floor(x: float, qmin: float, step: float) -> float:
    """不超过 x 的最大合法股数:0,或者 qmin + k * step。"""
    if x < qmin - 1e-9:
        return 0.0
    return qmin + step * math.floor((x - qmin) / step + 1e-9)


def qty_floor_vec(x: np.ndarray, qmin: np.ndarray, step: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        k = np.floor((x - qmin) / step + 1e-9)
        return np.where(x >= qmin - 1e-9, qmin + step * k, 0.0)


def fill_prices(adj: np.ndarray, raw: np.ndarray, fac: np.ndarray) -> np.ndarray:
    """标签价缺失、但当天有成交(不复权价在)的格子,用 不复权价 x 复权因子 补上。
    复权因子当天算不出来(整天没有一个有 Vwap 的分钟)就沿用最近一天的
    —— 因子只在除权日变。"""
    f = np.where(np.isfinite(fac) & (fac > 0), fac, np.nan)
    f = np.where(np.isfinite(f), f, ffill_rows(f))
    return np.where(np.isfinite(adj), adj, raw * f)


def load_market(path: str, fill: bool = True) -> Market:
    """fill=True:用不复权价 x 复权因子补齐标签价缺失的交易日(见 fill_prices)。"""
    z = np.load(path, allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    symbols = [str(s) for s in z["symbols"]]
    up, down = limit_locks(z["hi15"].astype(np.float64), z["lo15"].astype(np.float64),
                           z["close"].astype(np.float64), board_limit(symbols, dates))
    adj_label = z["adj"].astype(np.float64)
    raw = z["raw"].astype(np.float64)
    adj = fill_prices(adj_label, raw, z["fac"].astype(np.float64)) if (fill and "fac" in z.files) else adj_label
    qmin, qstep = lot_rules(symbols)
    return Market(dates=dates, adj=adj, raw=raw, vol15=z["vol15"].astype(np.float64), up_lock=up,
                  down_lock=down, symbols=symbols,
                  list_day=z["list_day"].astype(np.int64) if "list_day" in z.files else None,
                  lot_min=qmin, lot_step=qstep, adj_label=adj_label)


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------

TOPUP_IDLE = 0.01        # max_names:名额满了还闲着超过这一比例的现金就按排名补仓(见 simulate)
TOPUP_SLACK = 0.1        # 补仓最多补到单票上限的 1.1 倍:吸收整手取整留下的零头(每只平均差半手)
TOPUP_MIN = 0.2          # 补仓额不到单票上限的这一比例就不补:小单主要在付 5 元最低佣金


def simulate(mkt: Market, score: Optional[np.ndarray], t0: int, t1: int, *,
             capital: float = 1e6, cap: Optional[float] = 0.02, lot: int = LOT,
             pool_frac: float = 0.2, weights: Optional[np.ndarray] = None,
             costs: Costs = FLAT15, participation: Optional[float] = 0.1, band: float = 0.0,
             compound: bool = True, tradability: bool = True,
             reserve: float = 0.002, min_age: int = 0, floor: float = 0.0,
             conviction: Optional[Tuple[float, float]] = None, skip_top: float = 0.0,
             keep: float = 0.0, keep_units: bool = False, max_names: int = 0) -> Dict[str, np.ndarray]:
    """逐日模拟一个真账户。决策日 t ∈ [t0, t1):第 t 天收盘后用 score[t](或 weights[t])
    定目标,在 t+1 开盘 15 分钟按 VWAP 成交,到 t+2 同一时点结算当天收益。

    记账用"后复权单位":市值 = 单位数 x 后复权价,所以除权除息不会让市值跳;
    当前不复权股数 = 市值 / 不复权价,手数和单票上限都在这个空间里判断。

    capital    初始资金(元)。compound=False 时每天开盘前把净值重置回 capital
               (多退少补),训练段选参数用,让资金规模不随前几年的盈亏漂走。
    cap        单票上限(占当日净值);None = 不设上限。只约束新建和加仓,
               已有持仓涨过上限、只要超出 1 手以上就会被减回来。
    lot        0 = 股数可以是任意实数(理想口径);否则按 mkt.lot_min / lot_step 的交易单位
               (没有就一律 lot 股一手)。
    band       已有持仓离目标不到 band x 目标金额就不动(省掉追着漂移的小单);0 = 不设。
    reserve    建仓时留出的现金比例,给手续费用。
    tradability False 时忽略涨跌停和成交量上限(停牌仍然没法成交:没有价格)。
    min_age    上市不足这么多个交易日的不进候选池(先剔掉再排名)、不新买不加仓,
               已有持仓照常卖;0 = 不限。需要 mkt.list_day。

    下面四个只对 greedy(给 score)起作用:
    floor      单票下限(占净值):轮到某只票时分到的钱不到 floor 就不开仓;0 = 不设。
    conviction (z_lo, z_hi):目标仓位随信号强度在 [floor, cap] 之间线性变化 ——
               z 是当天合格股票里截面标准化后的信号,z <= z_lo 给 floor,z >= z_hi 给 cap。
               None = 每只都给 cap(等权)。
    skip_top   排名最前的这一比例不新买(信号最极端的头部,见 docs/可交易回测.md);0 = 不剔。
    keep       老持仓保留区:排名还在前 keep 的持仓原样留着、先占预算(超出上限 band 以上才减),
               剩下的钱再按排名补新票;0 = 不保留,每天按排名从头分配。
    max_names  最多持有几只(0 = 不限):卖不动的持仓和保留区里的老持仓先占名额,剩下的名额按排名补;
               名额满了,排名再往后的不开新仓、已有的清掉。名额满了还闲着超过 TOPUP_IDLE 的现金时(保留区、
               不调区间让持仓漂到上限以下,整手取整也留零头,钱又不能去买第 N+1 只),按排名把已选的票补到
               单票上限的 1 + TOPUP_SLACK 倍,这一轮不看不调区间,但不到 TOPUP_MIN x 上限的小单不做;加仓量本身
               要是合法股数(科创板 200 股起)。否则限只数的组合会长期拿着一成左右的现金,比的就成了仓位而不是只数。
               这是软上限:成交量上限让该清的票当天没卖完时,持股数会暂时超过 N。

    返回逐日数组:t、ret(净值收益)、nav(开盘前净值)、turnover(成交金额/2/净值)、
    fees(费用/净值)、n_hold(收盘持股只数)、cash(成交后现金/净值)、
    frozen(停牌持仓按最后价格计的市值/净值 —— 退市前停牌的票会一直挂在这里)、
    blocked_buy(想买但封涨停的只数)、blocked_sell(停牌或封跌停、动不了的持仓只数);
    keep_units=True 时另给 units(每天成交后的后复权单位数,(T, S))。"""
    if (score is None) == (weights is None):
        raise ValueError("score 和 weights 必须且只能给一个")
    panel = weights if weights is not None else score
    D, S = panel.shape
    if mkt.adj.shape != (D, S):
        raise ValueError(f"行情 {mkt.adj.shape} 和信号 {panel.shape} 形状不一致")
    if min_age and mkt.list_day is None:
        raise ValueError("min_age 需要 Market.list_day(重跑 prepare)")
    t_last = min(t1, D - 2)            # 决策日 t 要 t+2 的成交价才算得出当天收益
    if lot:
        qmin = mkt.lot_min if mkt.lot_min is not None else np.full(S, float(lot))
        qstep = mkt.lot_step if mkt.lot_step is not None else np.full(S, float(lot))
    else:
        qmin = qstep = np.zeros(S)
    qmin_l, qstep_l = qmin.tolist(), qstep.tolist()
    units = np.zeros(S)
    last = np.full(S, np.nan)
    cash = float(capital)
    cap_frac = float(cap) if cap else math.inf
    if conviction is not None and not math.isfinite(cap_frac):
        raise ValueError("conviction 需要有限的 cap")
    if floor and floor > cap_frac:
        raise ValueError(f"floor {floor} 不能大于 cap {cap}")
    rec: Dict[str, list] = {k: [] for k in ("t", "ret", "nav", "turnover", "fees", "n_hold", "cash",
                                            "frozen", "blocked_buy", "blocked_sell")}
    prev_nav: Optional[float] = None

    for t in range(t0, t_last + 1):
        e = t + 1
        px, rp = mkt.adj[e], mkt.raw[e]
        priced = np.isfinite(px) & np.isfinite(rp) & (rp > 0)
        last = np.where(priced, px, last)
        held = units > 0
        value = np.zeros(S)
        value[held] = units[held] * last[held]
        nav = cash + float(value.sum())
        if prev_nav is not None:
            rec["ret"].append(nav / prev_nav - 1.0)
        if t == t_last:
            break
        if not compound:
            cash += capital - nav
            nav = float(capital)
        date = mkt.dates[e]

        if tradability:
            can_buy = priced & ~mkt.up_lock[e]
            can_sell = priced & ~mkt.down_lock[e]
            if participation:
                max_sh = np.nan_to_num(participation * mkt.vol15[e], nan=0.0)
            else:
                max_sh = np.full(S, np.inf)
        else:
            can_buy = can_sell = priced
            max_sh = np.full(S, np.inf)
        old_enough = (e - mkt.list_day >= min_age) if min_age else np.ones(S, dtype=bool)
        can_buy = can_buy & old_enough
        h_sh = np.zeros(S)
        h_sh[priced] = value[priced] / rp[priced]
        forced = held & ~can_sell                  # 卖不动的持仓:原样留着,先占预算
        target = h_sh.copy()
        target[held & can_sell] = 0.0              # 能卖的默认清掉,进了候选再按目标改写
        remaining = nav * (1.0 - reserve) - float(value[forced].sum())

        blocked_buy = 0
        rp_l, h_l = rp.tolist(), h_sh.tolist()
        pr_l, cb_l, fo_l = priced.tolist(), can_buy.tolist(), forced.tolist()
        handled = np.zeros(S, dtype=bool)
        if weights is None:
            sc = score[t]
            fin = np.isfinite(sc) & old_enough     # 次新股是可交易池的事:先剔掉再排名,不占前 20% 的名额
            n_fin = int(fin.sum())
            ranked = np.argsort(np.where(fin, -sc, np.inf), kind="stable")[:n_fin]
            pct = np.full(S, np.inf)               # 排名分位,第 1 名接近 0
            pct[ranked] = (np.arange(n_fin) + 0.5) / max(n_fin, 1)
            n_pool = max(1, int(np.floor(n_fin * pool_frac))) if n_fin else 0
            order = ranked[:n_pool]
            if skip_top:
                order = order[pct[order] > skip_top]
            if conviction is not None:
                z_lo, z_hi = conviction
                v_all = sc[fin]
                sd = float(v_all.std()) if n_fin > 1 else 0.0
                z = (sc[order] - float(v_all.mean())) / (sd if sd > 0 else 1.0)
                conv = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)
                want = (floor + (cap_frac - floor) * conv) * nav
            else:
                want = np.full(order.size, cap_frac * nav)
            if keep:
                # 老持仓保留区:排名还在前 keep 的持仓原样留着,只在超出上限 band 以上时减回上限,
                # 先占预算;剩下的钱再按排名去补新票。换手主要花在"排名在边界上进进出出"上
                for i in np.flatnonzero(held & can_sell & (pct <= keep)).tolist():
                    p, h = rp_l[i], h_l[i]
                    qm, qs = qmin_l[i], qstep_l[i]
                    sh = h
                    cap_v = cap_frac * nav
                    if h * p > cap_v * (1.0 + band):
                        d = (cap_v / p) - h
                        sh = h + math.copysign(qty_floor(abs(d), qm, qs), d) if lot else cap_v / p
                    remaining -= sh * p
                    target[i] = sh
                    handled[i] = True
        else:
            w = weights[t]
            order = np.flatnonzero(np.isfinite(w) & (w > 0))
            order = order[np.argsort(-w[order], kind="stable")]
            want = np.minimum(w[order], cap_frac) * nav
        han_l = handled.tolist()
        floor_v = floor * nav
        slots = math.inf
        if max_names and weights is None:
            slots = max_names - int(forced.sum()) - int((handled & (target > 0)).sum())
        for i, wv in zip(order.tolist(), want.tolist()):
            if fo_l[i] or not pr_l[i] or han_l[i]:
                continue
            if remaining <= 0 or slots <= 0:
                break
            p, h = rp_l[i], h_l[i]
            qm, qs = qmin_l[i], qstep_l[i]
            v = min(wv, remaining)
            if v < floor_v * (1 - 1e-9):           # 分到的钱不到下限:不开这个仓(已有的也清掉)
                continue
            sh = qty_floor(v / p, qm, qs) if lot else v / p
            if not cb_l[i] and sh > h:
                blocked_buy += 1
                sh = h
            if h > 0 and sh > 0:
                if band > 0 and abs(sh - h) * p < band * wv and h * p <= remaining:
                    sh = h
                elif lot and sh != h:
                    # 增减量本身要是合法股数(清仓才允许卖零股),不够一笔就不动 —— 现金分红在
                    # 后复权记账下会让股数带小数尾巴(1003.2 股),不能为这点尾巴卖掉一整手
                    d = sh - h
                    sh = h + math.copysign(qty_floor(abs(d), qm, qs), d)
            remaining -= sh * p
            target[i] = sh
            if sh > 0:
                slots -= 1
        if max_names and weights is None and slots <= 0 and remaining > TOPUP_IDLE * nav and math.isfinite(cap_frac):
            cap_v = cap_frac * nav
            lim_v = cap_v * (1.0 + TOPUP_SLACK)
            min_v = TOPUP_MIN * cap_v
            for i in order.tolist():
                if remaining < min_v:
                    break
                if fo_l[i] or not pr_l[i] or not cb_l[i] or target[i] <= 0:
                    continue
                p, h = rp_l[i], h_l[i]
                room = min(lim_v - target[i] * p, remaining)
                if room < min_v:
                    continue
                want_sh = target[i] + room / p
                if lot:
                    qm, qs = qmin_l[i], qstep_l[i]
                    if h > 0 and want_sh >= h:         # 相对现有持仓买进:增量往下取到合法股数
                        new_sh = h + qty_floor(want_sh - h, qm, qs)
                    elif h > 0:                        # 还是要卖(只是少卖一点):卖出量往上取到合法股数,不超空间
                        need = h - want_sh
                        q = qty_floor(need, qm, qs)
                        if q < need - 1e-9:
                            q = q + qs if q >= qm else qm
                        new_sh = max(h - q, 0.0)
                    else:
                        new_sh = qty_floor(want_sh, qm, qs)
                else:
                    new_sh = want_sh
                add = new_sh - target[i]
                if add <= 0 or add * p < min_v or add * p > room + 1e-6:
                    continue
                target[i] = new_sh
                remaining -= add * p

        # 成交量上限:超出的部分这次不做(卖单也一样,剩下的明天再说)
        delta = np.where(priced, target - h_sh, 0.0)
        over = np.abs(delta) > max_sh + 1e-9
        if over.any():
            allowed = qty_floor_vec(max_sh[over], qmin[over], qstep[over]) if lot else max_sh[over]
            delta[over] = np.sign(delta[over]) * allowed
        sells = np.flatnonzero(delta < -1e-9)
        s_not = -delta[sells] * rp[sells]
        s_fee = costs.many(s_not, True, date)
        cash += float(s_not.sum() - s_fee.sum())
        units[sells] += delta[sells] * rp[sells] / px[sells]
        gone = sells[(h_sh[sells] + delta[sells]) <= 1e-6]
        units[gone] = 0.0                          # 清仓:别留浮点尾巴

        b_not = b_fee = 0.0
        buy_rate = costs.buy_rate()
        for i in order.tolist():
            d = delta[i]
            if d <= 1e-9:
                continue
            p = rp_l[i]
            n = d * p
            f = costs.one(n, False, date)
            if n + f > cash:                       # 钱不够:按交易单位往下减
                if lot:
                    qm, qs = qmin_l[i], qstep_l[i]
                    q = qty_floor(min(d, max(0.0, cash - costs.min_commission) / (p * (1 + buy_rate))), qm, qs)
                    while q > 0 and q * p + costs.one(q * p, False, date) > cash:
                        q = qty_floor(q - qs, qm, qs)
                    n = q * p
                else:
                    n = max(0.0, (cash - costs.min_commission) / (1 + buy_rate))
                f = costs.one(n, False, date)
            if n <= 0:
                continue
            cash -= n + f
            units[i] += n / px[i]
            b_not += n
            b_fee += f

        fees = float(s_fee.sum()) + b_fee
        rec["t"].append(t)
        rec["nav"].append(nav)
        rec["turnover"].append((float(s_not.sum()) + b_not) / (2.0 * nav))
        rec["fees"].append(fees / nav)
        rec["n_hold"].append(int((units > 0).sum()))
        rec["cash"].append(cash / max(nav - fees, 1e-12))
        rec["frozen"].append(float(value[held & ~priced].sum()) / nav)
        rec["blocked_buy"].append(blocked_buy)
        rec["blocked_sell"].append(int(forced.sum()))      # 停牌 + 封跌停的持仓
        if keep_units:
            rec.setdefault("units", []).append(units.copy())
        prev_nav = nav
    n = len(rec["ret"])
    return {k: np.asarray(v[:n] if k != "ret" else v, dtype=np.float64) for k, v in rec.items()}


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def stats(r: np.ndarray) -> Dict[str, float]:
    """和 long_only_grid.stats 同口径(年化 = 均值 x 252,回撤按日收益累加),
    另外给复利口径的年化(cagr)和回撤(mdd_c)。"""
    f = np.asarray(r, dtype=np.float64)
    f = f[np.isfinite(f)]
    if f.size < 3 or f.std() == 0:
        return {"sharpe": np.nan, "ann": np.nan, "mdd": np.nan, "cagr": np.nan, "mdd_c": np.nan}
    eq = np.cumsum(f)
    nav = np.cumprod(1.0 + f)
    return {"sharpe": float(f.mean() / f.std() * np.sqrt(252)),
            "ann": float(f.mean() * 252),
            "mdd": float((eq - np.maximum.accumulate(eq)).min()),
            "cagr": float(nav[-1] ** (252.0 / f.size) - 1.0),
            "mdd_c": float((nav / np.maximum.accumulate(nav) - 1.0).min())}


def summarize(res: Dict[str, np.ndarray], bench: np.ndarray, capital: float) -> Dict[str, float]:
    r = res["ret"]
    ex = r - bench[res["t"].astype(int)]
    a, x = stats(r), stats(ex)
    return {"ex_sharpe": x["sharpe"], "ex_ann": x["ann"], "ex_mdd": x["mdd"],
            "abs_sharpe": a["sharpe"], "abs_ann": a["ann"], "cagr": a["cagr"], "abs_mdd_c": a["mdd_c"],
            "final": float(capital * np.prod(1.0 + r)), "turnover": float(res["turnover"].mean()),
            "fees_yr": float(res["fees"].mean() * 252), "n_hold": float(res["n_hold"].mean()),
            "cash": float(res["cash"].mean()), "frozen_end": float(res["frozen"][-1]) if r.size else np.nan,
            "frozen_max": float(res["frozen"].max()) if r.size else np.nan,
            "blocked_buy": float(res["blocked_buy"].mean()),
            "blocked_sell": float(res["blocked_sell"].mean()), "days": int(r.size)}


# ---------------------------------------------------------------------------
# prepare:服务器上从立方体算行情面板
# ---------------------------------------------------------------------------

def prepare(store: str, start: str, end: str, out: str, chunk: int = 256) -> int:
    from alphacube.label import daily_close, open_vwap15
    from alphacube.research import warmed_view
    from alphacube.store import DAY_STATUS_CODES, open_cube_store

    t0 = time.time()
    view, _ = warmed_view(open_cube_store(store), start, end, warmup=60)   # 和网格/信号同一个视图
    D, S = view.n_days, view.n_symbols
    print(f"{view.dates[0]}..{view.dates[-1]}: {D} 天 x {S} 只", flush=True)
    adj = open_vwap15(view)
    print(f"  后复权 VWAP15 ({time.time() - t0:.0f}s)", flush=True)
    close = daily_close(view)
    print(f"  收盘价 ({time.time() - t0:.0f}s)", flush=True)
    lo_m, hi_m = OPEN_WINDOW
    raw = np.full((D, S), np.nan)
    vol15 = np.full((D, S), np.nan)
    hi15 = np.full((D, S), np.nan)
    lo15 = np.full((D, S), np.nan)
    fac = np.full((D, S), np.nan)
    import warnings
    for lo in range(0, S, chunk):
        hi = min(lo + chunk, S)
        amt_all = view.read_s("Amt", lo, hi)
        vol_all = view.read_s("Volume", lo, hi)
        vw_all = view.read_s("Vwap", lo, hi)
        # 当天的复权因子 = 后复权价 / 不复权价,一天之内每分钟都一样;按成交量加权,
        # 用全天所有 Vwap 在的分钟(科创板开盘 15 分钟常常只剩两三分钟有 Vwap)
        ok = np.isfinite(vw_all) & np.isfinite(amt_all) & np.isfinite(vol_all) & (vol_all > 0) & (amt_all > 0)
        num = np.where(ok, vw_all.astype(np.float64) * vol_all, 0.0).sum(-1)
        den = np.where(ok, amt_all.astype(np.float64), 0.0).sum(-1)
        with np.errstate(all="ignore"):
            fac[:, lo:hi] = np.where(den > 0, num / den, np.nan).T
        del vw_all, ok, num, den
        amt = amt_all[..., lo_m:hi_m].astype(np.float64)
        vol = vol_all[..., lo_m:hi_m].astype(np.float64)
        del amt_all, vol_all
        good = np.isfinite(amt) & np.isfinite(vol) & (vol > 0) & (amt > 0)
        A = np.where(good, amt, 0.0).sum(-1)
        V = np.where(good, vol, 0.0).sum(-1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw[:, lo:hi] = np.where(V > 0, A / V, np.nan).T
            vol15[:, lo:hi] = V.T
            hi15[:, lo:hi] = np.nanmax(view.read_s("High", lo, hi)[..., lo_m:hi_m], axis=-1).T
            lo15[:, lo:hi] = np.nanmin(view.read_s("Low", lo, hi)[..., lo_m:hi_m], axis=-1).T
        print(f"  {hi}/{S} ({time.time() - t0:.0f}s)", flush=True)
    bad = ~view.tradable().T
    for a in (raw, vol15, hi15, lo15, fac):
        a[bad] = np.nan
    # 上市日:第一个进入生命周期(交易/复牌/停牌)的交易日。窗口第一天就在生命周期里的,
    # 是窗口开始前上市的老股,记成一个很小的负数 —— 不能用"第一个有价格的日子",
    # 那会把 2018 年初恰好停牌的老股当成次新股
    status = view.store.read_status(view._rows(0, S), view.d0, view.d1)
    alive = status != DAY_STATUS_CODES["OUTSIDE_LIFECYCLE"]
    list_day = np.where(alive.any(1), np.argmax(alive, axis=1), D).astype(np.int64)
    list_day[alive[:, 0]] = -10_000
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, dates=np.array(view.dates), symbols=np.array(view.symbols), adj=adj, list_day=list_day,
             raw=raw.astype(np.float32), vol15=vol15.astype(np.float32), hi15=hi15.astype(np.float32),
             lo15=lo15.astype(np.float32), close=close.astype(np.float32), fac=fac)
    print(f"写出 {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# run:复现基线 -> 扫 cap x 半衰期(训练段选)-> 报测试段
# ---------------------------------------------------------------------------

_G: dict = {}


def _grid_cell(args):
    hl, cap, min_age = args
    m, sm, bench, split, D, kw = _G["mkt"], _G["smoothed"][hl], _G["bench"], _G["split"], _G["D"], _G["kw"]
    tr = simulate(m, sm, 0, split, cap=cap, compound=False, min_age=min_age, **kw)
    te = simulate(m, sm, split, D, cap=cap, compound=True, min_age=min_age, **kw)
    return hl, cap, min_age, summarize(tr, bench, kw["capital"]), summarize(te, bench, kw["capital"])


def _fmt_cap(c: Optional[float]) -> str:
    return "无上限" if not c or c >= 1 else f"{c:.1%}"


def run(a: argparse.Namespace) -> int:
    from long_only_grid import long_only_weights, rank_order
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows, portfolio_returns

    t_start = time.time()
    mkt = load_market(a.inputs)
    sig = np.load(a.signal).astype(np.float64)
    D, S = sig.shape
    if mkt.adj.shape != (D, S):
        raise SystemExit(f"行情 {mkt.adj.shape} 和信号 {sig.shape} 对不上")
    def ew_bench(price: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = forward_return(price, 1, 1)
        with np.errstate(all="ignore"):
            fin = np.isfinite(lab)
            return lab, np.where(fin.any(1), np.nansum(lab, 1) / np.maximum(fin.sum(1), 1), np.nan)

    # 标签口径(补价之前)复现基线;补价之后的基准给真账户那几行用 —— 两边都是等权全市场,
    # 只差科创板那些标签缺失的股票日有没有算进来
    label, bench0 = ew_bench(mkt.adj_label)
    _, bench = ew_bench(mkt.adj)
    split = max(1, min(D - 1, int(D * a.train_frac)))
    test = slice(split, D)
    cost = COST_MODELS[a.cost]
    print(f"{mkt.dates[0]}..{mkt.dates[-1]}  {D} 天 x {S} 只;测试段从 {mkt.dates[split]} 起;"
          f"初始资金 {a.capital:,.0f} 元;费用 {cost.name}")
    up, dn = mkt.up_lock[test], mkt.down_lock[test]
    pr = np.isfinite(mkt.adj[test])
    print(f"测试段开盘 15 分钟:封涨停 {up.sum() / pr.sum():.2%} 封跌停 {dn.sum() / pr.sum():.2%} "
          f"(占有价的股票日)")
    filled = np.isfinite(mkt.adj) & ~np.isfinite(mkt.adj_label)
    print(f"补齐的成交价:{filled.sum() / max(np.isfinite(mkt.adj).sum(), 1):.2%} 的有价股票日"
          f"(其中科创板占 {filled[:, mkt.lot_min == 200].sum() / max(filled.sum(), 1):.0%})")
    bench_test = bench0[test]
    out: Dict[str, object] = {"inputs": a.inputs, "signal": a.signal, "capital": a.capital,
                              "cost": cost.name, "split_date": mkt.dates[split]}
    rows: List[Tuple[str, Dict[str, float]]] = []

    # (0) 原网格口径:权重回测,复现 +1.92
    hl0 = a.base_hl
    sm0 = ema_rows(sig, hl0) if hl0 else sig
    w0 = long_only_weights(sm0, 0.2, 0, "exp1", cached=rank_order(sm0))
    net, _, turn = portfolio_returns(w0[test], label[test], 0.0015)
    x0, a0 = stats(net - bench_test), stats(net)
    ref = {"ex_sharpe": x0["sharpe"], "ex_ann": x0["ann"], "ex_mdd": x0["mdd"], "abs_sharpe": a0["sharpe"],
           "abs_ann": a0["ann"], "cagr": a0["cagr"], "abs_mdd_c": a0["mdd_c"],
           "final": float(a.capital * np.prod(1 + net[np.isfinite(net)])), "turnover": float(np.nanmean(turn)),
           "fees_yr": float(np.nanmean(turn) * 2 * 0.0015 * 252), "n_hold": float((w0[test] > 0).sum(1).mean()),
           "cash": 0.0, "frozen_end": 0.0, "blocked_buy": 0.0, "blocked_sell": 0.0,
           "days": int(np.isfinite(net).sum())}
    rows.append((f"0 原网格:前20%·exp1·hl{hl0:g},权重口径,单边15bp", ref))
    print(f"[{time.time() - t_start:.0f}s] 基线复现 超额夏普 {ref['ex_sharpe']:+.2f}", flush=True)

    import dataclasses
    base_kw = dict(capital=a.capital, participation=a.participation, reserve=0.0)
    unfilled = dataclasses.replace(mkt, adj=mkt.adj_label)
    r = simulate(unfilled, None, split, D, weights=w0, cap=None, lot=0, costs=FLAT15, tradability=False,
                 **base_kw)
    rows.append(("1 同上,换成真账户记账(净值复利、持仓随价漂移)", summarize(r, bench0, a.capital)))
    r = simulate(mkt, None, split, D, weights=w0, cap=None, lot=0, costs=FLAT15, tradability=False,
                 **base_kw)
    rows.append(("2 + 补齐科创板缺失的开盘价(基准同步补齐)", summarize(r, bench, a.capital)))
    r = simulate(mkt, None, split, D, weights=w0, cap=None, lot=0, costs=FLAT15, tradability=True,
                 **base_kw)
    rows.append(("3 + 涨跌停买卖不了、成交量上限 10%", summarize(r, bench, a.capital)))
    r = simulate(mkt, None, split, D, weights=w0, cap=None, lot=LOT, costs=FLAT15, tradability=True,
                 **base_kw)
    rows.append((f"4 + 整手({a.capital / 1e4:,.0f} 万元,把前 20% 的权重直接取整)", summarize(r, bench, a.capital)))
    print(f"[{time.time() - t_start:.0f}s] 权重口径几行完成", flush=True)

    # (4)(5) 用户目标:greedy,扫 cap x 半衰期,训练段(资金恒定)选,测试段(复利)报。
    # 两套可交易池各扫一遍:全部股票 / 上市不足 min_age 天的不买,各自在训练段选参数
    caps = [float(c) for c in a.caps.split(",")]
    hls = [float(h) for h in a.halflives.split(",")]
    ages = [0, a.min_age] if a.min_age else [0]
    smoothed = {hl: (ema_rows(sig, hl) if hl else sig) for hl in hls}
    kw = dict(capital=a.capital, lot=LOT, pool_frac=a.pool, costs=cost, participation=a.participation,
              band=a.band, tradability=True, reserve=a.reserve)
    _G.update(mkt=mkt, smoothed=smoothed, bench=bench, split=split, D=D, kw=kw)
    cells = [(hl, c, ag) for ag in ages for hl in hls for c in caps]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            grid = pool.map(_grid_cell, cells)
    else:
        grid = [_grid_cell(c) for c in cells]
    print(f"[{time.time() - t_start:.0f}s] 网格 {len(cells)} 格完成", flush=True)

    def train_best(ag):
        cand = [g for g in grid if g[2] == ag]
        return max(cand, key=lambda g: g[3]["ex_sharpe"] if np.isfinite(g[3]["ex_sharpe"]) else -9e9)

    chosen = {ag: train_best(ag) for ag in ages}
    b0 = chosen[0]
    rows.append((f"5 用户目标(greedy) cap {_fmt_cap(b0[1])}·hl{b0[0]:g}(训练段选),整手", b0[4]))
    head_age = ages[-1]
    bh = chosen[head_age]
    if head_age:
        rows.append((f"6 + 上市不足 {head_age} 天不买(重选:cap {_fmt_cap(bh[1])}·hl{bh[0]:g})", bh[4]))
    bhl, bcap = bh[0], bh[1]
    sm_b = smoothed[bhl]
    hkw = {**kw, "cap": bcap, "min_age": head_age}
    if not cost.min_commission:
        # 有最低佣金时这一行没有意义:零碎股数每天都在微调,每笔都按最低佣金收
        v = simulate(mkt, sm_b, split, D, **{**hkw, "lot": 0})
        rows.append(("  a 同上,但股数可以是任意实数(隔离整手的影响)", summarize(v, bench, a.capital)))
    other = "retail" if a.cost == "flat15" else "flat15"
    v = simulate(mkt, sm_b, split, D, **{**hkw, "costs": COST_MODELS[other]})
    rows.append((f"  b 同上,费用换成 {COST_MODELS[other].name}", summarize(v, bench, a.capital)))
    if not a.band:
        v = simulate(mkt, sm_b, split, D, **{**hkw, "band": 0.25})
        rows.append(("  c 同上,已有持仓偏离目标不到 25% 不调", summarize(v, bench, a.capital)))

    hdr = (f"{'':<50} {'超额夏普':>7} {'超额年化':>7} {'超额回撤':>7} | {'绝对夏普':>7} {'复利年化':>7} "
           f"{'复利回撤':>7} {'期末资金':>10} | {'换手':>6} {'费用/年':>7} {'持股':>5} {'现金':>6} {'停牌挂账':>7}")
    print("\n=== 测试段(" + f"{mkt.dates[split]} 放入 {a.capital:,.0f} 元)===")
    print(hdr)
    for name, r in rows:
        print(f"{name:<50} {r['ex_sharpe']:+7.2f} {r['ex_ann']:+7.1%} {r['ex_mdd']:+7.3f} | "
              f"{r['abs_sharpe']:+7.2f} {r['cagr']:+7.1%} {r['abs_mdd_c']:+7.3f} {r['final']:>10,.0f} | "
              f"{r['turnover']:6.1%} {r['fees_yr']:7.2%} {r['n_hold']:5.0f} {r['cash']:6.1%} "
              f"{r.get('frozen_end', 0.0):7.2%}")

    for ag in ages:
        pick = chosen[ag]
        print(f"\n=== cap x 半衰期,{'全部股票' if not ag else f'上市不足 {ag} 天不买'}"
              f"(训练段:资金恒定 {a.capital:,.0f};测试段:复利)===")
        print(f"{'cap':>6} {'hl':>4} | {'训练超额夏普':>10} | {'测试超额夏普':>10} {'超额年化':>7} {'超额回撤':>7} "
              f"{'绝对夏普':>7} {'期末资金':>10} {'换手':>6} {'持股':>5} {'现金':>6} {'买不进':>6}")
        for hl, c, g_ag, tr, te in sorted((g for g in grid if g[2] == ag), key=lambda g: (g[1], g[0])):
            mark = " <- 训练段最优" if (hl, c) == (pick[0], pick[1]) else ""
            print(f"{_fmt_cap(c):>6} {hl:4g} | {tr['ex_sharpe']:+10.2f} | {te['ex_sharpe']:+10.2f} "
                  f"{te['ex_ann']:+7.1%} {te['ex_mdd']:+7.3f} {te['abs_sharpe']:+7.2f} {te['final']:>10,.0f} "
                  f"{te['turnover']:6.1%} {te['n_hold']:5.0f} {te['cash']:6.1%} {te['blocked_buy']:6.1f}{mark}")

    out["rows"] = [{"name": n, **r} for n, r in rows]
    out["grid"] = [{"halflife": hl, "cap": c, "min_age": ag, "train": tr, "test": te}
                   for hl, c, ag, tr, te in grid]
    out["chosen"] = {str(ag): {"halflife": g[0], "cap": g[1]} for ag, g in chosen.items()}
    head = simulate(mkt, sm_b, split, D, **hkw)
    out["headline_daily"] = {"dates": [mkt.dates[int(t)] for t in head["t"]], "ret": head["ret"].tolist(),
                             "bench": bench[head["t"].astype(int)].tolist(), "nav": head["nav"].tolist(),
                             "n_hold": head["n_hold"].tolist(), "turnover": head["turnover"].tolist(),
                             "cash": head["cash"].tolist(), "fees": head["fees"].tolist()}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
        print(f"\n写出 {a.out}")
    print(f"({time.time() - t_start:.0f}s)")
    return 0


def _sweep_cell(args):
    name, hl, kw = args
    m, bench, split, D, base = _G["mkt"], _G["bench"], _G["split"], _G["D"], _G["base"]
    sm = _G["smoothed"][hl]
    tr = simulate(m, sm, _G.get("t0", 0), split, compound=False, **{**base, **kw})
    te = simulate(m, sm, split, D, compound=True, **{**base, **kw})
    return name, hl, kw, summarize(tr, bench, base["capital"]), summarize(te, bench, base["capital"])


def _run_cells(cells, workers):
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_sweep_cell, cells)
    return [_sweep_cell(c) for c in cells]


def _print_cells(title, res, best, top=None):
    print(f"\n=== {title}(按训练段超额夏普排序)===")
    print(f"{'构建':<44} {'hl':>3} | {'训练':>6} | {'测试超额夏普':>8} {'超额年化':>7} {'超额回撤':>7} {'绝对夏普':>7} "
          f"{'期末资金':>10} {'换手':>6} {'费用/年':>7} {'持股':>5} {'现金':>6}")
    rows = sorted(res, key=lambda r: -(r[3]["ex_sharpe"] if np.isfinite(r[3]["ex_sharpe"]) else -9e9))
    for r in rows[: top or len(rows)]:
        name, hl, kw, tr, te = r
        mark = " <-" if r is best else ""
        print(f"{name:<44} {hl:3g} | {tr['ex_sharpe']:+6.2f} | {te['ex_sharpe']:+8.2f} {te['ex_ann']:+7.1%} "
              f"{te['ex_mdd']:+7.3f} {te['abs_sharpe']:+7.2f} {te['final']:>10,.0f} {te['turnover']:6.1%} "
              f"{te['fees_yr']:7.2%} {te['n_hold']:5.0f} {te['cash']:6.1%}{mark}")


def sweep(a: argparse.Namespace) -> int:
    """两轮实验,每轮都只在训练段选:
    1) 怎么省手续费:等权,cap x 不调区间 band x 老持仓保留区 keep x 半衰期;
    2) 仓位怎么分:用第 1 轮选出的 band/keep/半衰期,比较
       等权(不同 cap)/ 按信号强度在 [floor, cap] 之间分(conviction)/ 剔除最极端的头部(skip_top)。"""
    from alphacube.label import forward_return
    from alphacube.metrics import ema_rows

    t_start = time.time()
    mkt = load_market(a.inputs)
    sig = np.load(a.signal).astype(np.float64)
    D, S = sig.shape
    lab = forward_return(mkt.adj, 1, 1)
    universe = None
    if a.universe:
        from universe import load_universe
        universe = load_universe(a.universe, (D, S))
        lab = np.where(universe, lab, np.nan)            # 基准 = 池子里等权
    with np.errstate(all="ignore"):
        fin = np.isfinite(lab)
        bench = np.where(fin.any(1), np.nansum(lab, 1) / np.maximum(fin.sum(1), 1), np.nan)
    split = max(1, min(D - 1, int(D * a.train_frac)))
    cost = COST_MODELS[a.cost]
    hls = [float(h) for h in a.halflives.split(",")]
    base = dict(capital=a.capital, lot=LOT, pool_frac=a.pool, costs=cost, participation=a.participation,
                tradability=True, reserve=a.reserve, min_age=a.min_age, max_names=a.max_names)
    smoothed = {hl: ema_rows(sig, hl) for hl in hls}
    if universe is not None:
        for hl in hls:                                   # 先平滑再圈池子:平滑可以用池外那段历史
            smoothed[hl][~universe] = np.nan
    t0 = 0
    if a.train_start:                                    # 滚动信号前几百天是空的:训练段从有信号的那天算起
        t0 = next((i for i, x in enumerate(mkt.dates) if str(x) >= a.train_start), split)
        if t0 >= split:
            raise SystemExit(f"--train-start {a.train_start} 不在训练段里")
    _G.update(mkt=mkt, bench=bench, split=split, D=D, base=base, smoothed=smoothed, t0=t0)
    print(f"训练段从 {mkt.dates[t0]} 起;测试段从 {mkt.dates[split]} 起;初始资金 {a.capital:,.0f} 元;费用 {cost.name};"
          f"上市不足 {a.min_age} 天不买;候选池 = 信号前 {a.pool:.0%}"
          f"{';股票池 ' + a.universe + '(基准 = 池内等权)' if universe is not None else ''}", flush=True)

    # ---- 第 1 轮:省手续费
    floats = lambda txt: [float(x) for x in txt.split(",")]
    caps1, bands, keeps, skips = floats(a.caps1), floats(a.bands), floats(a.keeps), floats(a.skips)
    cells = []
    for cap in caps1:
        for band in bands:
            for keep in keeps:
                for hl in hls:
                    name = f"等权 {cap:.1%} 不调{band:.0%} 保留{'前' + format(keep, '.0%') if keep else '关'}"
                    cells.append((name, hl, dict(cap=cap, band=band, keep=keep)))
    r1 = _run_cells(cells, a.workers)
    b1 = max(r1, key=lambda r: r[3]["ex_sharpe"] if np.isfinite(r[3]["ex_sharpe"]) else -9e9)
    print(f"[{time.time() - t_start:.0f}s] 第 1 轮 {len(cells)} 格", flush=True)
    _print_cells("第 1 轮:省手续费(等权)", r1, b1, top=a.top)
    ref = [] if a.max_names else [r for r in r1 if r[2] == dict(cap=0.005, band=0.0, keep=0.0) and r[1] == 10.0]
    if ref:
        _print_cells("对照:上一版(等权 0.5%,不调 0,不保留,hl10)", ref, None)
    else:
        print("\n(网格里没有上一版那一格:等权 0.5%,不调 0,不保留,hl10)")
    hl1 = b1[1]
    print(f"\n第 1 轮训练段最优:{b1[0]} hl{hl1:g}。各 band/保留区在 hl{hl1:g}、cap {b1[2]['cap']:.1%} 下"
          f"(训练 / 测试 超额夏普,测试费用/年):")
    print(f"{'':>10}" + "".join(f"{'不调' + format(b, '.0%'):>24}" for b in bands))
    for keep in keeps:
        line = f"{('保留前' + format(keep, '.0%')) if keep else '不保留':>10}"
        for band in bands:
            r = [x for x in r1 if x[1] == hl1 and x[2] == dict(cap=b1[2]["cap"], band=band, keep=keep)][0]
            line += f"   {r[3]['ex_sharpe']:+5.2f} / {r[4]['ex_sharpe']:+5.2f} {r[4]['fees_yr']:5.1%}"
        print(line)

    def pack(res):
        return [{"name": n, "halflife": hl, "kw": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                 "train": tr, "test": te} for n, hl, kw, tr, te in res]

    if a.stage1_only:
        if a.out:
            Path(a.out).write_text(json.dumps({"cost": cost.name, "capital": a.capital, "universe": a.universe,
                                               "stage1": pack(r1), "best1": b1[0], "best1_hl": b1[1],
                                               "best1_kw": b1[2]}, ensure_ascii=False, indent=1, default=float))
            print(f"\n写出 {a.out}")
        print(f"({time.time() - t_start:.0f}s)")
        return 0

    # ---- 第 2 轮:仓位怎么分
    fee = dict(band=b1[2]["band"], keep=b1[2]["keep"])
    cells = []
    for cap in (0.003, 0.005, 0.01, 0.02):
        cells.append((f"等权 {cap:.1%}", hl1, dict(cap=cap, **fee)))
    for floor, cap in ((0.0025, 0.01), (0.0025, 0.02), (0.005, 0.01), (0.005, 0.02), (0.005, 0.03),
                       (0.01, 0.02), (0.01, 0.03)):
        for z_hi in (2.0, 3.0, 4.0):
            cells.append((f"强度分仓 [{floor:.2%},{cap:.0%}] z1→{z_hi:g}", hl1,
                          dict(cap=cap, floor=floor, conviction=(1.0, z_hi), **fee)))
    for cap in (0.005, 0.01):
        for skip in skips:
            cells.append((f"等权 {cap:.1%} 剔除前{skip:.1%}", hl1, dict(cap=cap, skip_top=skip, **fee)))
    r2 = _run_cells(cells, a.workers)
    b2 = max(r2, key=lambda r: r[3]["ex_sharpe"] if np.isfinite(r[3]["ex_sharpe"]) else -9e9)
    print(f"\n[{time.time() - t_start:.0f}s] 第 2 轮 {len(cells)} 格(band {fee['band']:.0%},"
          f"保留区 {fee['keep']:.0%},hl{hl1:g})", flush=True)
    _print_cells("第 2 轮:仓位怎么分", r2, b2)

    if a.out:
        Path(a.out).write_text(json.dumps({"cost": cost.name, "capital": a.capital, "stage1": pack(r1),
                                           "stage2": pack(r2), "best1": b1[0], "best2": b2[0]},
                                          ensure_ascii=False, indent=1, default=float))
        print(f"\n写出 {a.out}")
    print(f"({time.time() - t_start:.0f}s)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--store", default="/root/autodl-tmp/qmt_daily_data/fast_factor_compute")
    p.add_argument("--start", default="20180102")
    p.add_argument("--end", default="20251231")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--signal", required=True)
    r.add_argument("--capital", type=float, default=1e6)
    r.add_argument("--cost", choices=sorted(COST_MODELS), default="flat15")
    r.add_argument("--caps", default="0.001,0.002,0.003,0.005,0.01,0.02,0.05,0.1,1.0")
    r.add_argument("--halflives", default="2,3,5,10,20")
    r.add_argument("--base-hl", type=float, default=5.0, help="基线(前20%·exp1)的半衰期,原网格选出的是 5")
    r.add_argument("--pool", type=float, default=0.2, help="greedy 只在信号前这么多比例里分配")
    r.add_argument("--participation", type=float, default=0.1)
    r.add_argument("--band", type=float, default=0.0)
    r.add_argument("--min-age", type=int, default=60, help="上市不足这么多个交易日的不买;0 = 不限")
    r.add_argument("--reserve", type=float, default=0.002)
    r.add_argument("--train-frac", type=float, default=0.7)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", default="")
    w = sub.add_parser("sweep", help="省手续费 / 仓位分配两轮实验,训练段选")
    w.add_argument("--inputs", required=True)
    w.add_argument("--signal", required=True)
    w.add_argument("--capital", type=float, default=1e6)
    w.add_argument("--cost", choices=sorted(COST_MODELS), default="retail")
    w.add_argument("--halflives", default="5,10,20")
    w.add_argument("--caps1", default="0.005,0.01", help="第 1 轮的单票上限")
    w.add_argument("--bands", default="0,0.1,0.25,0.5", help="第 1 轮的不调区间")
    w.add_argument("--keeps", default="0,0.1,0.2,0.3", help="第 1 轮的老持仓保留区(0 = 不保留)")
    w.add_argument("--skips", default="0.001,0.002,0.005,0.01", help="第 2 轮剔除头部的比例")
    w.add_argument("--universe", default="", help="股票池掩码 (D, S) bool(scripts/universe.py),基准换成池内等权")
    w.add_argument("--stage1-only", action="store_true", help="只跑第 1 轮(单票上限 x 不调区间 x 保留区 x 半衰期)")
    w.add_argument("--train-start", default="", help="训练段从这天(YYYYMMDD)起算;滚动拟合的信号前面是空的")
    w.add_argument("--max-names", type=int, default=0, help="最多持有几只(0 = 不限);一般配 --caps1 1/N")
    w.add_argument("--pool", type=float, default=0.2)
    w.add_argument("--participation", type=float, default=0.1)
    w.add_argument("--reserve", type=float, default=0.002)
    w.add_argument("--min-age", type=int, default=60)
    w.add_argument("--train-frac", type=float, default=0.7)
    w.add_argument("--workers", type=int, default=10)
    w.add_argument("--top", type=int, default=20)
    w.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a.store, a.start, a.end, a.out)
    if a.cmd == "sweep":
        return sweep(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
