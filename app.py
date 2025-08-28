
# app.py — DayTrade App Pro + Watchlist (expert scoring, robust fetch, fallback, caching)
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App Pro", layout="wide")
st.title("📊 DayTrade App Pro — Screener, Watchlist, Backtest, Live")

# ===================== Config =====================
DEFAULT_CFG = {
    "screener_equities": {"tickers": ["AAPL","MSFT","TSLA","NVDA","AMD","META","AMZN","GOOGL","SMCI","NFLX","QQQ","SPY"],
        "lookback_days": 120, "min_price": 3, "min_avg_volume": 1_000_000, "top_n": 10,
        "score_weights": {"momentum":0.45,"volatility":0.2,"liquidity":0.2,"trend_quality":0.15},
        "atr_mult_target": 2.5},
    "screener_crypto": {"tickers": ["BTC-USD","ETH-USD","SOL-USD","AVAX-USD","DOGE-USD","XRP-USD","ADA-USD","LINK-USD"],
        "lookback_days": 120, "top_n": 10,
        "score_weights": {"momentum":0.55,"volatility":0.25,"liquidity":0.15,"trend_quality":0.05},
        "atr_mult_target": 3.0},
    "backtest": {"tickers": ["AAPL","MSFT","TSLA"], "lookback_days": 10, "or_minutes": 15,
        "ema_period": 20, "atr_period": 14, "atr_mult": 2.0, "allow_short": True,
        "starting_capital": 100000, "risk_per_trade": 0.005, "end_liquidate": "15:55"},
    "watchlist": {"tickers": ["AAPL","NVDA","TSLA","SPY","QQQ","BTC-USD","ETH-USD"], "interval": "1h", "lookback_days": 90}
}

def _merge_cfg(user_cfg, defaults):
    out = dict(defaults)
    for k, v in (user_cfg or {}).items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge_cfg(v, out[k])
        else:
            out[k] = v
    return out

def load_config():
    try:
        if os.path.exists("config.yaml"):
            with open("config.yaml","r") as f:
                user = yaml.safe_load(f) or {}
            return _merge_cfg(user, DEFAULT_CFG)
        return DEFAULT_CFG
    except Exception:
        return DEFAULT_CFG

cfg = load_config()
if not os.path.exists("config.yaml"):
    st.info("ℹ️ `config.yaml` non trouvé — j'utilise la configuration par défaut.")

# ===================== Utils robustes =====================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")

def s_open(df):   return safe_column(df,"Open")
def s_high(df):   return safe_column(df,"High")
def s_low(df):    return safe_column(df,"Low")
def s_close(df):  return safe_column(df,"Close")
def s_volume(df): return safe_column(df,"Volume")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = s_high(df), s_low(df), s_close(df)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    c, v = s_close(df), s_volume(df)
    pv = (c*v).cumsum()
    vv = v.cumsum().replace(0, np.nan)
    return pv / vv

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)

def nz(series: pd.Series, default=0.0):
    if series is None or len(series) == 0:
        return float(default)
    return finite(series.iloc[-1], default)

def max_days_for_interval(interval: str) -> int:
    interval = str(interval).lower()
    if interval == "1m": return 7
    if interval in {"2m","5m","15m","30m","90m","60m"}: return 60
    if interval in {"1h","2h","4h"}: return 730
    return 3650

def synthetic_df(days: int, interval: str, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(int(days*8), 120)
    rets = rng.normal(0.0005, 0.02, n)
    px = np.cumprod(1 + rets) * start_price
    high = px * (1 + rng.normal(0.001, 0.002, n).clip(-0.01, 0.02))
    low  = px * (1 - rng.normal(0.001, 0.002, n).clip(-0.02, 0.01))
    open_ = (high + low) / 2
    vol = np.abs(rng.normal(1e7, 2e6, n)).astype(int)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="H" if "m" in interval else "D")
    df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":px, "AdjClose":px, "Volume":vol}, index=idx)
    return df

@st.cache_data(show_spinner=False, ttl=300)
def fetch(symbol: str, lookback_days: int, interval: str, buffer_days: int = 40) -> pd.DataFrame:
    max_days = max_days_for_interval(interval)
    period_days = int(min(max(lookback_days + buffer_days, 10), max_days))
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False, auto_adjust=False, prepost=False, threads=True)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        base = 30000.0 if "-USD" in symbol else 100.0
        df = synthetic_df(lookback_days, interval, start_price=base)
    df = flatten_cols(df)
    if "Adj Close" in df.columns: df = df.rename(columns={"Adj Close":"AdjClose"})
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df.columns: df[c] = np.nan
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","AdjClose","Volume"]].copy()

# ===================== Scoring helpers =====================
def adaptive_momentum(close: pd.Series) -> float:
    n = len(close)
    if n >= 22 and pd.notna(close.iloc[-21]) and close.iloc[-21] != 0:
        return float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21])
    if n >= 6 and pd.notna(close.iloc[-5]) and close.iloc[-5] != 0:
        return float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5])
    return 0.0

def trend_quality_score(close: pd.Series) -> float:
    n = len(close)
    e20 = nz(ema(close,20), np.nan)
    e50 = nz(ema(close,50), np.nan) if n >= 50 else np.nan
    e200 = nz(ema(close,200), np.nan) if n >= 200 else np.nan
    if np.isfinite(e20) and np.isfinite(e50) and np.isfinite(e200):
        return 1.0 if (e20 > e50 > e200) else (0.5 if (e20 > e50 or e50 > e200) else 0.0)
    if np.isfinite(e20) and np.isfinite(e50):
        return 0.7 if e20 > e50 else 0.3
    return 0.5

def percentile_ranks(values, higher_is_better=True):
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    pct = s.rank(pct=True, ascending=not higher_is_better).values
    return pct

def build_screener_rows(symbols, interval, lookback, weights, atr_mult_target, is_crypto=False, force_topn=True):
    rows = []
    raw = []
    for sym in symbols:
        df = fetch(sym, lookback_days=lookback, interval=interval, buffer_days=40 if not is_crypto else 30)
        close = s_close(df)
        if len(close) < 3 or close.isna().all():
            raw.append({"symbol": sym, "price": 0.0, "momentum": 0.0, "atr_pct": 0.0, "liq": 0.0, "tq": 0.5, "atr_last": 0.0})
            continue
        price = nz(close, 0.0)
        mom = adaptive_momentum(close)
        atr_s = atr(df, 14)
        atr_last = nz(atr_s, 0.0)
        atr_pct = float(atr_last / price) if price > 0 else 0.0
        vol30 = s_volume(df).tail(30)
        if not vol30.isna().all() and vol30.sum() > 0 and price > 0:
            avg_dvol = float((close.tail(30) * vol30).dropna().mean() or 0.0)
            liq = float(np.log10(max(avg_dvol, 1.0)) / 8.0)
        else:
            liq = float(np.clip(atr_pct * 2.0, 0.0, 1.0)) if is_crypto else 0.0
        tq = trend_quality_score(close)
        raw.append({"symbol": sym, "price": price, "momentum": mom, "atr_pct": atr_pct, "liq": liq, "tq": tq, "atr_last": atr_last})

    mom_p = percentile_ranks([r["momentum"] for r in raw], higher_is_better=True)
    vol_p = percentile_ranks([r["atr_pct"] for r in raw], higher_is_better=True)
    liq_p = percentile_ranks([r["liq"] for r in raw], higher_is_better=True)
    tq_v  = np.array([r["tq"] for r in raw], dtype=float)

    for i, r in enumerate(raw):
        score = (
            weights.get("momentum",0.5)      * float(mom_p[i]) +
            weights.get("volatility",0.2)    * float(vol_p[i]) +
            weights.get("liquidity",0.2)     * float(liq_p[i]) +
            weights.get("trend_quality",0.1) * float(tq_v[i])
        )
        potential = float(atr_mult_target * r["atr_last"])
        rows.append({
            "symbol": r["symbol"],
            "price": float(r["price"]),
            "score": float(score),
            "momentum": float(r["momentum"]),
            "atr_pct": float(r["atr_pct"]),
            "liquidity_score": float(r["liq"]),
            "trend_quality": float(r["tq"]),
            "potential_gain_$": float(potential)
        })
    return rows

# ===================== Signals (ORB + EMA + VWAP + ATR-stop) =====================
def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = pd.to_datetime(df.index)
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def first_true_index(cond) -> pd.Timestamp | None:
    if isinstance(cond, pd.DataFrame):
        cond = cond.any(axis=1)
    if isinstance(cond, pd.Series):
        cond = cond.fillna(False)
        idx = cond[cond].index
        return idx[0] if len(idx) else None
    return None

def generate_signals(df: pd.DataFrame, ema_period=20, atr_period=14, atr_mult=2.0, or_minutes=15, allow_short=True):
    if df.empty:
        return df
    df = df.copy()
    df = flatten_cols(df)
    df["Open"], df["High"], df["Low"] = s_open(df), s_high(df), s_low(df)
    df["Close"], df["Volume"] = s_close(df), s_volume(df)
    df["EMA"], df["ATR"], df["VWAP"] = ema(df["Close"], ema_period), atr(df, atr_period), vwap(df)
    df["session"] = pd.to_datetime(df.index).normalize()
    for c in ["direction","entry","stop","or_high","or_low"]:
        df[c] = None

    for session, g in df.groupby("session"):
        mask_or = compute_opening_range_mask(g, minutes=or_minutes)
        if mask_or.sum() == 0:
            continue
        or_high = float(g.loc[mask_or, "High"].max())
        or_low  = float(g.loc[mask_or, "Low"].min())
        after = g.loc[~mask_or]
        if after.empty or not np.isfinite(or_high) or not np.isfinite(or_low):
            continue
        long_cond  = (after["Close"] > or_high) & (after["EMA"] > after["VWAP"]) & (after["Volume"] > 0)
        short_cond = (after["Close"] < or_low)  & (after["EMA"] < after["VWAP"]) & (after["Volume"] > 0)
        e_long  = first_true_index(long_cond)
        e_short = first_true_index(short_cond) if allow_short else None

        entry_idx, direction = None, None
        if e_long is not None and (e_short is None or e_long <= e_short):
            entry_idx, direction = e_long, "long"
        elif e_short is not None:
            entry_idx, direction = e_short, "short"
        if entry_idx is None:
            continue

        entry   = float(after.at[entry_idx, "Close"])
        atr_val = float(after.at[entry_idx, "ATR"])
        stop = max(or_low, entry - atr_mult * atr_val) if direction == "long" else min(or_high, entry + atr_mult * atr_val)
        ix = g.index[g.index >= entry_idx]
        df.loc[ix, ["direction","entry","stop","or_high","or_low"]] = [direction, entry, stop, or_high, or_low]
    return df

def pos_size(capital, entry, stop, risk_pct):
    if entry is None or stop is None or entry <= 0:
        return 0
    rps = abs(entry - stop)
    if rps <= 0:
        return 0
    return int((capital * risk_pct) // rps)

# ===================== UI =====================
tab_scr, tab_watch, tab_bt, tab_live = st.tabs(["🔎 Screener", "⭐ Watchlist", "🧪 Backtest", "📡 Live"])

# -------- Screener --------
with tab_scr:
    st.subheader("Screener (Actions / Crypto) — scoring multi-facteurs")
    capital = st.number_input("Capital (pour sizing indicatif)", value=100000.0, step=1000.0)
    risk_pct = st.number_input("Risque par trade (ex: 0.005 = 0.5%)", value=0.005, step=0.001, format="%.3f")
    mode = st.radio("Mode", ["Equities (actions)", "Crypto"], horizontal=True)
    force_topn = st.checkbox("Forcer un Top N (tolérant aux données manquantes)", value=True)

    if mode.startswith("Equities"):
        sc = cfg.get("screener_equities", {})
        symbols = st.multiselect("Tickers (actions)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","1h","30m","15m","5m"], index=0)
        lookback = st.slider("Lookback (jours)", 30, 365, sc.get("lookback_days", 120))
        min_price = st.number_input("Prix minimum", value=float(sc.get("min_price", 3)))
        min_avg_vol = st.number_input("Volume moyen min (actions)", value=float(sc.get("min_avg_volume", 1_000_000)))
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10))
        weights = sc.get("score_weights", {"momentum":0.45,"volatility":0.2,"liquidity":0.2,"trend_quality":0.15})
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 2.5)))

        if st.button("Lancer (actions)") and symbols:
            rows_all = build_screener_rows(symbols, interval, lookback, weights, atr_mult_target, is_crypto=False, force_topn=force_topn)
            df = pd.DataFrame(rows_all)
            if not force_topn:
                filtered = []
                for sym in symbols:
                    raw = fetch(sym, lookback_days=lookback, interval=interval)
                    price = nz(s_close(raw), 0.0)
                    avg_vol = float(s_volume(raw).tail(30).mean() or 0.0)
                    if price >= min_price and avg_vol >= min_avg_vol:
                        r = df[df["symbol"]==sym].iloc[0].to_dict()
                        filtered.append(r)
                df = pd.DataFrame(filtered) if filtered else df

            # stop & sizing indicatif
            df["rec_stop"], df["rec_size"] = 0.0, 0
            for i, sym in enumerate(df["symbol"]):
                data = fetch(sym, lookback_days=lookback, interval=interval)
                c = s_close(data); a = atr(data, 14)
                entry = nz(c, 0.0); stop = entry - 2.0 * nz(a, 0.0)
                df.loc[df.index[i], "rec_stop"] = stop
                df.loc[df.index[i], "rec_size"] = pos_size(capital, entry, stop, risk_pct)

            out = df.sort_values("score", ascending=False).head(int(topn))
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Exporter (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="screener_equities.csv", mime="text/csv")

    else:
        sc = cfg.get("screener_crypto", {})
        symbols = st.multiselect("Tickers (crypto)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m","5m"], index=0, key="cr_int")
        lookback = st.slider("Lookback (jours)", 30, 365, sc.get("lookback_days", 120), key="cr_lb")
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10), key="cr_top")
        weights = sc.get("score_weights", {"momentum":0.55,"volatility":0.25,"liquidity":0.15,"trend_quality":0.05})
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 3.0)), key="cr_atr")

        if st.button("Lancer (crypto)", key="cr_run") and symbols:
            df = pd.DataFrame(build_screener_rows(symbols, interval, lookback, weights, atr_mult_target, is_crypto=True, force_topn=force_topn))
            # stop & sizing indicatif
            df["rec_stop"], df["rec_size"] = 0.0, 0
            for i, sym in enumerate(df["symbol"]):
                data = fetch(sym, lookback_days=lookback, interval=interval)
                c = s_close(data); a = atr(data, 14)
                entry = nz(c, 0.0); stop = entry - 2.5 * nz(a, 0.0)
                df.loc[df.index[i], "rec_stop"] = stop
                df.loc[df.index[i], "rec_size"] = pos_size(capital, entry, stop, risk_pct)

            out = df.sort_values("score", ascending=False).head(int(topn))
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Exporter (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="screener_crypto.csv", mime="text/csv")

# -------- Watchlist --------
with tab_watch:
    st.subheader("⭐ Watchlist — signaux rapides & alertes")
    wc = cfg.get("watchlist", {})
    wl_tickers = st.text_area("Liste (séparés par des virgules)", ", ".join(wc.get("tickers", []))).strip()
    wl_symbols = [s.strip() for s in wl_tickers.split(",") if s.strip()] if wl_tickers else wc.get("tickers", [])
    wl_interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m"], index=2)
    wl_lookback = st.slider("Lookback (jours)", 30, 365, wc.get("lookback_days", 90))
    show_rsi = st.checkbox("Afficher RSI(14)", value=True)
    show_ema = st.checkbox("Afficher EMA20/50/200", value=True)
    run_watch = st.button("Mettre à jour watchlist")

    if run_watch and wl_symbols:
        cards = []
        for sym in wl_symbols:
            df = fetch(sym, lookback_days=wl_lookback, interval=wl_interval, buffer_days=30)
            if df.empty:
                continue
            c = s_close(df)
            price = nz(c, 0.0)
            a = atr(df, 14); atr_last = nz(a, 0.0)
            e20 = ema(c,20); e50 = ema(c,50); e200 = ema(c,200)
            vw = vwap(df)
            # Signaux
            r = rsi(c,14) if show_rsi else None
            ema_bull = (nz(e20,np.nan) > nz(e50,np.nan) > nz(e200,np.nan))
            breakout_20 = price >= float(c.tail(20).max())
            pullback_e20 = price >= nz(e20, price) and (price - nz(e20, price)) / max(price,1e-9) <= 0.02  # <=2% au-dessus EMA20
            volat = float(atr_last / price) if price>0 else 0.0
            alerts = []
            if ema_bull: alerts.append("Tendance haussière (EMA20>50>200)")
            if breakout_20: alerts.append("Breakout 20-bar")
            if pullback_e20: alerts.append("Pullback sain vers EMA20")
            if volat >= 0.04: alerts.append("Volatilité élevée (ATR%≥4%)")
            if show_rsi and r is not None:
                rv = nz(r, 50.0)
                if rv < 35: alerts.append("RSI survendu")
                elif rv > 65: alerts.append("RSI suracheté")

            cards.append({
                "symbol": sym, "price": price, "ATR%": round(volat*100,2),
                "EMA20": float(nz(e20, np.nan)), "EMA50": float(nz(e50, np.nan)), "EMA200": float(nz(e200, np.nan)),
                "VWAP": float(nz(vw, np.nan)),
                "RSI14": float(nz(r, np.nan)) if show_rsi else np.nan,
                "Alerts": " • ".join(alerts) if alerts else "—"
            })
        if cards:
            dfw = pd.DataFrame(cards)
            st.dataframe(dfw, use_container_width=True)
            st.download_button("⬇️ Exporter watchlist (CSV)", data=dfw.to_csv(index=False).encode("utf-8"), file_name="watchlist.csv", mime="text/csv")
        else:
            st.info("Aucun symbole valide. Vérifie la liste.")

# -------- Backtest --------
with tab_bt:
    st.subheader("Backtest ORB + EMA20 + VWAP + Stop ATR")
    bcfg = cfg.get("backtest", {})
    tickers = st.multiselect("Tickers", bcfg.get("tickers", []), default=bcfg.get("tickers", []))
    interval = st.selectbox("Intervalle", ["5m","15m","30m","1h"], index=0, key="bt_int")
    lookback_days = st.slider("Historique (jours)", 5, 30, bcfg.get("lookback_days", 10), key="bt_days")
    or_minutes = st.slider("Opening Range (minutes)", 5, 30, bcfg.get("or_minutes", 15), key="bt_or")
    ema_p = st.number_input("EMA period", value=int(bcfg.get("ema_period", 20)), key="bt_ema")
    atr_p = st.number_input("ATR period", value=int(bcfg.get("atr_period", 14)), key="bt_atr")
    atr_mult = st.number_input("ATR × (stop)", value=float(bcfg.get("atr_mult", 2.0)), key="bt_mult")
    allow_short = st.checkbox("Autoriser short", value=bool(bcfg.get("allow_short", True)), key="bt_short")
    end_liq = st.text_input("Liquidation (HH:MM)", bcfg.get("end_liquidate", "15:55"), key="bt_liq")
    runbt = st.button("Lancer le backtest")

    if runbt and tickers:
        summary = []
        for sym in tickers:
            df = fetch(sym, lookback_days=lookback_days, interval=interval, buffer_days=20)
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)

            pnl = 0.0
            trades = 0
            h, m = map(int, end_liq.split(":"))
            end_t = time(hour=h, minute=m)
            wins = 0
            losses = 0
            rs = []

            for session, g in sig.groupby("session"):
                rows = g[g["entry"].notna()]
                if rows.empty:
                    continue
                entry_ts = rows.index[0]
                direction = rows.iloc[0]["direction"]
                entry = rows.iloc[0]["entry"]
                stop = rows.iloc[0]["stop"]
                risk_per_share = abs(entry - stop)
                trades += 1
                exited = False
                for ts, row in g.loc[g.index >= entry_ts].iterrows():
                    if ts.time() >= end_t:
                        price = row["Close"]
                        r = (price - entry)/risk_per_share if direction=="long" else (entry - price)/risk_per_share
                        pnl += (price - entry) if direction=="long" else (entry - price)
                        rs.append(r); wins += 1 if r > 0 else 0; losses += 1 if r <= 0 else 0
                        exited = True; break
                    if direction == "long":
                        if row["Low"] <= stop:
                            r = (stop - entry)/risk_per_share; pnl += (stop - entry)
                            rs.append(r); losses += 1; exited = True; break
                    else:
                        if row["High"] >= stop:
                            r = (entry - stop)/risk_per_share; pnl += (entry - stop)
                            rs.append(r); losses += 1; exited = True; break
                    a = row["ATR"]
                    stop = max(stop, row["Close"] - atr_mult * a) if direction == "long" else min(stop, row["Close"] + atr_mult * a)
                if not exited:
                    last = g.iloc[-1]["Close"]
                    r = (last - entry)/risk_per_share if direction=="long" else (entry - last)/risk_per_share
                    pnl += (last - entry) if direction=="long" else (entry - last)
                    rs.append(r); wins += 1 if r > 0 else 0; losses += 1 if r <= 0 else 0

            winrate = (wins / max(trades,1)) * 100.0
            avg_r = float(np.mean(rs)) if rs else 0.0
            max_dd = float(np.min(np.cumsum(rs))) if rs else 0.0
            summary.append({"symbol": sym, "trades": trades, "winrate_%": winrate, "avg_R": avg_r, "cum_R": float(np.sum(rs)), "approx_maxDD_R": max_dd})

        if summary:
            st.dataframe(pd.DataFrame(summary).sort_values("cum_R", ascending=False), use_container_width=True)
        else:
            st.info("Pas de signaux sur cette période.")

# -------- Live --------
with tab_live:
    st.subheader("Live (simulation) — signaux intraday")
    st.caption("Aperçu éducatif — aucune exécution réelle.")
    live_symbol = st.text_input("Ticker", "AAPL")
    interval = st.selectbox("Intervalle (intraday)", ["5m","15m","30m"], index=0, key="live_int")
    days = st.slider("Jours à afficher", 1, 5, 1, key="live_days")
    ema_p = st.number_input("EMA period", value=20, key="live_ema")
    atr_p = st.number_input("ATR period", value=14, key="live_atr")
    atr_mult = st.number_input("ATR × (stop)", value=2.0, key="live_mult")
    or_minutes = st.slider("Opening Range (min)", 5, 30, 15, key="live_or")
    allow_short = st.checkbox("Autoriser short", value=True, key="live_short")
    runlive = st.button("Actualiser")
    if runlive and live_symbol:
        df = fetch(live_symbol, lookback_days=days, interval=interval, buffer_days=10)
        if df.empty:
            st.warning("Pas de données.")
        else:
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)
            cols = [c for c in ["Close","EMA","VWAP"] if c in sig.columns]
            if len(cols) >= 2 and sig[cols].dropna().shape[0] >= 5:
                st.line_chart(sig[cols], use_container_width=True)
            else:
                st.info("Pas assez de points propres pour tracer.")
            cols_last = [c for c in ["direction","entry","stop","or_high","or_low"] if c in sig.columns]
            if cols_last:
                st.table(sig.iloc[-1:][cols_last])
