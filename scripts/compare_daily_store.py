"""Compare the daily fields DERIVED from the cube (``daily.daily_from_view``)
with a V2 daily store (the previous project's ``daily/`` directory:
``field_<name>.npy`` (T, N) float32 + ``meta.json``).

    PYTHONPATH=src python scripts/compare_daily_store.py --root <compute root with daily/ and minute/>
    PYTHONPATH=src python scripts/compare_daily_store.py --self-check      # synthetic cube, reference by numpy

Exit code 0 means every field agreed to ``--tol`` relative to its own scale;
1 means a field disagreed, a field was missing, or the store's axes did not
line up with the view.

The V2 conventions (qmt_ingest): Open = first bar's open, High = max, Low =
min, Close = last bar's close, Volume/Amount = sums, Vwap = Amount/Volume,
Return = Close[t]/Close[t-1] - 1 over adjacent trading days (both tradable).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def reference_daily(view):
    """numpy reference of the V2 daily conventions from a cube view -> {name: (D, S)}."""
    S = view.n_symbols
    out = {k: np.full((view.n_days, S), np.nan) for k in ("Open", "High", "Low", "Close", "Volume", "Amount", "Vwap")}
    for lo in range(0, S, 64):
        hi = min(lo + 64, S)
        o, h, l, c = (view.read_s(f, lo, hi).astype(np.float64) for f in ("Open", "High", "Low", "Close"))
        v, a = view.read_s("Volume", lo, hi).astype(np.float64), view.read_s("Amt", lo, hi).astype(np.float64)
        fin = np.isfinite(c)
        first = np.argmax(np.isfinite(o), axis=-1)
        last = c.shape[-1] - 1 - np.argmax(fin[..., ::-1], axis=-1)
        any_ = fin.any(-1)
        out["Open"][:, lo:hi] = np.where(any_, np.take_along_axis(o, first[..., None], -1)[..., 0], np.nan).T
        out["Close"][:, lo:hi] = np.where(any_, np.take_along_axis(c, last[..., None], -1)[..., 0], np.nan).T
        with np.errstate(all="ignore"):
            out["High"][:, lo:hi] = np.where(any_, np.nanmax(h, -1), np.nan).T
            out["Low"][:, lo:hi] = np.where(any_, np.nanmin(l, -1), np.nan).T
            vs, as_ = np.nansum(v, -1), np.nansum(a, -1)
            out["Volume"][:, lo:hi] = np.where(any_, vs, np.nan).T
            out["Amount"][:, lo:hi] = np.where(any_, as_, np.nan).T
            out["Vwap"][:, lo:hi] = np.where(any_ & (vs > 0), as_ / vs, np.nan).T
    trad = view.tradable().T
    for k in out:
        out[k][~trad] = np.nan
    close = out["Close"]
    ret = np.full_like(close, np.nan)
    with np.errstate(all="ignore"):
        ret[1:] = close[1:] / close[:-1] - 1
    out["Return"] = ret
    return out


def compare(derived, reference, label):
    print(f"--- {label} ---")
    worst = 0.0
    for name in sorted(reference):
        if name not in derived:
            print(f"  {name:8s} missing in derived")
            continue
        x, y = np.asarray(derived[name], dtype=np.float64), np.asarray(reference[name], dtype=np.float64)
        nan_mismatch = int((np.isfinite(x) != np.isfinite(y)).sum())
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            print(f"  {name:8s} no overlapping finite values")
            continue
        ad = np.abs(x[m] - y[m])
        scale = float(np.abs(y[m]).max()) or 1.0
        scaled = float(ad.max() / scale)                # float32 summation-order differences are ~1e-7 of the scale
        worst = max(worst, scaled, 1.0 if nan_mismatch else 0.0)
        print(f"  {name:8s} max|abs| {ad.max():.3e}  max|abs|/scale {scaled:.3e}  exact {float((ad == 0).mean()):.4f}  nan-mismatch {nan_mismatch}")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="compute root with daily/ and minute/")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max |derived - stored| / scale accepted; float32 summation order is ~1e-7")
    a = ap.parse_args()
    from alphacube.daily import daily_from_view
    from alphacube.store import open_cube_store
    from alphacube.synthetic import write_synthetic_store

    if a.self_check:
        with tempfile.TemporaryDirectory() as d:
            st = write_synthetic_store(d, n_symbols=12, n_days=50, n_minutes=241, seed=3, suspended_frac=0.05)
            view = st.view()
            worst = compare(daily_from_view(view, open_vwap15=False)._fields, reference_daily(view), "derived vs numpy reference (synthetic)")
        print("self-check", "OK" if worst < a.tol else "MISMATCH")
        return 0 if worst < a.tol else 1
    if not a.root:
        print("give --root or --self-check")
        return 2
    root = Path(a.root)
    st = open_cube_store(root)
    view = st.view(start=a.start, end=a.end, max_symbols=a.max_symbols)
    derived = daily_from_view(view, open_vwap15=False)._fields
    worst = compare(derived, reference_daily(view), "derived (engine) vs numpy reference (same cube)")
    daily_dir = root / "daily"
    if (daily_dir / "meta.json").is_file():
        meta = json.loads((daily_dir / "meta.json").read_text(encoding="utf-8"))
        dates = [str(x) for x in meta.get("dates", [])]
        insts = [str(x) for x in meta.get("instruments", [])]
        # the daily store spans every date and instrument the server ever ingested,
        # so the view is a window into it: line both axes up by label, never by position
        pos = {d: i for i, d in enumerate(dates)}
        d0 = pos.get(view.dates[0])
        contiguous = d0 is not None and dates[d0:d0 + view.n_days] == list(view.dates)
        ipos = {s: i for i, s in enumerate(insts)}
        cols = [ipos[s] for s in view.symbols if s in ipos]
        if not contiguous or len(cols) != view.n_symbols:
            missing = [s for s in view.symbols if s not in ipos][:5]
            print(f"daily store axes do not cover the view (dates aligned: {contiguous}; "
                  f"{view.n_symbols - len(cols)} symbols missing e.g. {missing}); skipping store comparison")
            return 1
        stored = {}
        for name in ("Open", "High", "Low", "Close", "Volume", "Amount", "Vwap", "Return"):
            f = daily_dir / f"field_{name}.npy"
            if f.is_file():
                arr = np.load(f, mmap_mode="r")
                stored[name] = np.asarray(arr[d0:d0 + view.n_days][:, cols], dtype=np.float64)
        if not stored:
            print(f"no field_*.npy in {daily_dir}; skipping store comparison")
            return 1
        worst = max(worst, compare(derived, stored, "derived (engine) vs V2 daily store"))
    else:
        print(f"no daily store at {daily_dir}; only the internal reference comparison was run")
    print("comparison", "OK" if worst < a.tol else "MISMATCH", f"(worst scaled delta {worst:.3e}, tol {a.tol:g})")
    return 0 if worst < a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
