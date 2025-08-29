
# app.py — DayTrade App Pro All-in-One (YF Strong + AI Signals)
import os, time, math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dtime
import yaml

st.set_page_config(page_title="DayTrade App Pro — All-in-One", layout="wide")
st.title("📊 DayTrade App Pro — All-in-One (YF Strong + 🤖 AI Signals)")

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
    "backtest": {"tickers": ["AAPL","MSFT","TSLA"], "lookback_days": 10, "or_minutes": 15, "ema_period": 20,
        "atr_period": 14, "atr_mult": 2.0, "allow_short": True, "starting_capital": 100000, "risk_per_trade": 0.005, "end_liquidate": "15:55"},
    "watchlist": {"tickers": ["AAPL","NVDA","TSLA","SPY","QQQ","BTC-USD","ETH-USD"], "interval": "1h", "lookback_days": 90},
    "ai": {"tickers": ["AAPL","NVDA","TSLA","SPY","BTC-USD","ETH-USD"], "interval": "15m", "lookback_days": 90,
           "train_epochs": 250, "lr": 0.2, "l2": 1e-4, "buy_threshold": 0.6, "sell_threshold": 0.4}
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

# ===================== Utils & Indicators =====================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame): col = col.iloc[:,0]
    return pd.to_numeric(col, errors="coerce")

def s_open(df):   return safe_column(df,"Open")
def s_high(df):   return safe_column(df,"High")
def s_low(df):    return safe_column(df,"Low")
def s_close(df):  return safe_column(df,"Close")
def s_volume(df): return safe_column(df,"Volume")

def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = s_high(df), s_low(df), s_close(df)
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    c, v = s_close(df), s_volume(df)
    pv = (c*v).cumsum(); vv = v.cumsum().replace(0, np.nan)
    return pv / vv

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)

def nz(series: pd.Series, default=0.0):
    if series is None or len(series)==0: return float(default)
    return finite(series.iloc[-1], default)

# ===================== Strong Yahoo Fetch =====================
def max_days_for_interval(interval:str)->int:
    interval = interval.lower()
    if interval == "1m": return 7
    if interval in {"2m","5m","15m","30m","90m","60m"}: return 60
    if interval in {"1h","2h","4h"}: return 730
    return 3650

DOWNGRADE_CHAIN = {
    "1m": ["2m","5m","15m","30m","1h","4h","1d"],
    "2m": ["5m","15m","30m","1h","4h","1d"],
    "5m": ["15m","30m","1h","4h","1d"],
    "15m":["30m","1h","4h","1d"],
    "30m":["1h","4h","1d"],
    "1h": ["4h","1d"],
    "4h": ["1d"],
    "1d": []
}

def normalize_df(df):
    if df is None or df.empty: return pd.DataFrame()
    df2 = flatten_cols(df.copy())
    if "Adj Close" in df2.columns: df2.rename(columns={"Adj Close":"AdjClose"}, inplace=True)
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df2.columns: df2[c]=np.nan
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index)
    return df2[["Open","High","Low","Close","AdjClose","Volume"]]

def synthetic_df(days:int, interval:str, start_price:float=100.0, seed:int=42):
    rng = np.random.default_rng(seed)
    n = max(int(days*8), 120)
    rets = rng.normal(0.0005, 0.02, n)
    px = np.cumprod(1 + rets) * start_price
    hi = px * (1 + rng.normal(0.001,0.002,n).clip(-0.01,0.02))
    lo = px * (1 - rng.normal(0.001,0.002,n).clip(-0.02,0.01))
    op = (hi+lo)/2
    vol = np.abs(rng.normal(1e7,2e6,n)).astype(int)
    freq = "H" if "m" in interval or "h" in interval else "D"
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq=freq)
    return pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":px,"AdjClose":px,"Volume":vol}, index=idx)

def strong_fetch(symbol:str, interval:str, lookback_days:int, backoff_base:float=0.8, retries:int=3, strategy_log:list=None):
    if strategy_log is None: strategy_log = []
    used_interval = interval
    limit = max_days_for_interval(used_interval)
    if lookback_days > limit:
        strategy_log.append(f"interval {used_interval} too small for {lookback_days}d; capping to {limit}d")
        lookback_days = limit
    period = f"{max(lookback_days,5)}d"
    ok = False; df = pd.DataFrame()

    STRATS = [
        ("download(period)", lambda: yf.download(symbol, period=period, interval=used_interval, progress=False, auto_adjust=False, prepost=False, threads=True)),
        ("history(period)",  lambda: yf.Ticker(symbol).history(period=period, interval=used_interval, auto_adjust=False, prepost=False)),
        ("download(start/end)", lambda: yf.download(symbol, start=(datetime.utcnow()-timedelta(days=lookback_days)).strftime("%Y-%m-%d"), end=datetime.utcnow().strftime("%Y-%m-%d"), interval=used_interval, progress=False, auto_adjust=False, prepost=False, threads=True)),
    ]

    for attempt in range(1, retries+1):
        for name, fn in STRATS:
            try:
                strategy_log.append(f"try {name}, attempt {attempt}")
                tmp = fn()
                tmp = normalize_df(tmp)
                if tmp is not None and not tmp.empty:
                    df = tmp; ok = True
                    strategy_log.append(f"✔ success with {name} (rows={len(df)})")
                    break
                else:
                    strategy_log.append(f"❌ empty with {name}")
            except Exception as e:
                strategy_log.append(f"❌ error {name}: {type(e).__name__}")
        if ok: break
        time.sleep(backoff_base * (2 ** (attempt-1)))
        chain = DOWNGRADE_CHAIN.get(used_interval.lower(), [])
        if chain:
            used_interval = chain[0]
            strategy_log.append(f"↓ downgrading interval to {used_interval} and retrying")
            limit = max_days_for_interval(used_interval)
            if lookback_days > limit:
                strategy_log.append(f"capping lookback {lookback_days}d -> {limit}d for {used_interval}")
                lookback_days = limit
        else:
            strategy_log.append("no more interval downgrade options")
    return df, used_interval, strategy_log

def fetch(symbol:str, interval:str, lookback_days:int, mode:str="real_or_fallback"):
    log = []
    if mode == "fallback_only":
        base = 30000.0 if "-USD" in symbol else 100.0
        df = synthetic_df(lookback_days, interval, base)
        log.append("mode=fallback_only → synthetic data")
        return df, interval, log
    df, used_int, log = strong_fetch(symbol, interval, lookback_days, strategy_log=log)
    if df is None or df.empty:
        if mode == "real_or_fallback":
            base = 30000.0 if "-USD" in symbol else 100.0
            df = synthetic_df(lookback_days, interval, base)
            log.append("fallback engaged → synthetic data")
        else:
            log.append("real_only → no data")
    return df, used_int, log

# ===================== AI model (numpy logistic regression) =====================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = s_close(df); v = s_volume(df)
    e20 = ema(c,20); e50 = ema(c,50); e200 = ema(c,200)
    a = atr(df,14); vw = vwap(df)
    macd_line, macd_sig, macd_hist = macd(c)

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = c.pct_change(1)
    feats["ret_3"] = c.pct_change(3)
    feats["ret_5"] = c.pct_change(5)
    feats["ema20_slope"] = e20.diff()
    feats["ema50_slope"] = e50.diff()
    feats["rsi14"] = rsi(c,14)
    feats["atr_pct"] = a / c.replace(0,np.nan)
    feats["macd"] = macd_line
    feats["macd_hist"] = macd_hist
    feats["dist_vwap"] = (c - vw) / c.replace(0,np.nan)
    feats["breakout20"] = c / c.rolling(20).max() - 1.0
    feats["drawup20"] = c / c.rolling(20).min() - 1.0
    feats = feats.replace([np.inf,-np.inf], np.nan).dropna()
    return feats

def standardize(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma==0, 1.0, sigma)
    Xz = (X - mu) / sigma
    return Xz, mu, sigma

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0/(1.0+np.exp(-z))

def train_logreg_gd(X, y, lr=0.2, epochs=250, l2=1e-4):
    n, k = X.shape
    w = np.zeros(k); b = 0.0
    for _ in range(int(epochs)):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = (X.T @ (p - y))/n + l2*w
        grad_b = float(np.sum(p - y))/n
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def rule_based_prob(df: pd.DataFrame):
    c = s_close(df)
    e20 = ema(c,20); e50=ema(c,50); e200=ema(c,200)
    a = atr(df,14); vw = vwap(df); r = rsi(c,14)
    macd_line, macd_sig, macd_hist = macd(c)
    price = nz(c, 0.0)

    votes = []
    votes.append(1 if nz(e20,np.nan) > nz(e50,np.nan) > nz(e200,np.nan) else 0)
    votes.append(1 if nz(macd_hist,0.0) > 0 else 0)
    rv = nz(r, 50.0)
    votes.append(1 if 45 <= rv <= 65 else (0 if rv < 35 else 0))  # neutre = 1, extreme = 0
    votes.append(1 if price >= float(c.tail(20).max()) else 0)
    votes.append(1 if price >= nz(vw, price) else 0)
    p = 0.3 + 0.4 * (sum(votes)/len(votes))
    return float(np.clip(p, 0.25, 0.75))

# ===================== Signals & Backtest helpers =====================
def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = pd.to_datetime(df.index)
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def first_true_index(cond) -> pd.Timestamp | None:
    if isinstance(cond, pd.DataFrame): cond = cond.any(axis=1)
    if isinstance(cond, pd.Series):
        cond = cond.fillna(False); idx = cond[cond].index
        return idx[0] if len(idx) else None
    return None

def generate_signals(df: pd.DataFrame, ema_period=20, atr_period=14, atr_mult=2.0, or_minutes=15, allow_short=True):
    if df.empty: return df
    df = df.copy(); df = flatten_cols(df)
    df["Open"], df["High"], df["Low"] = s_open(df), s_high(df), s_low(df)
    df["Close"], df["Volume"] = s_close(df), s_volume(df)
    df["EMA"], df["ATR"], df["VWAP"] = ema(df["Close"], ema_period), atr(df, atr_period), vwap(df)
    df["session"] = pd.to_datetime(df.index).normalize()
    for c in ["direction","entry","stop","or_high","or_low"]: df[c] = None

    for session, g in df.groupby("session"):
        mask_or = compute_opening_range_mask(g, minutes=or_minutes)
        if mask_or.sum() == 0: continue
        or_high = float(g.loc[mask_or, "High"].max()); or_low  = float(g.loc[mask_or, "Low"].min())
        after = g.loc[~mask_or]
        if after.empty or not np.isfinite(or_high) or not np.isfinite(or_low): continue
        long_cond  = (after["Close"] > or_high) & (after["EMA"] > after["VWAP"]) & (after["Volume"] > 0)
        short_cond = (after["Close"] < or_low)  & (after["EMA"] < after["VWAP"]) & (after["Volume"] > 0)
        e_long  = first_true_index(long_cond)
        e_short = first_true_index(short_cond) if allow_short else None
        entry_idx, direction = None, None
        if e_long is not None and (e_short is None or e_long <= e_short):
            entry_idx, direction = e_long, "long"
        elif e_short is not None:
            entry_idx, direction = e_short, "short"
        if entry_idx is None: continue
        entry   = float(after.at[entry_idx, "Close"]); atr_val = float(after.at[entry_idx, "ATR"])
        stop = max(or_low, entry - atr_mult * atr_val) if direction == "long" else min(or_high, entry + atr_mult * atr_val)
        ix = g.index[g.index >= entry_idx]
        df.loc[ix, ["direction","entry","stop","or_high","or_low"]] = [direction, entry, stop, or_high, or_low]
    return df

def pos_size(capital, entry, stop, risk_pct):
    if entry is None or stop is None or entry <= 0: return 0
    rps = abs(entry - stop); 
    if rps <= 0: return 0
    return int((capital * risk_pct) // rps)

# ===================== Sidebar: data mode =====================
st.sidebar.header("⚙️ Source de données")
data_mode = st.sidebar.selectbox("Mode de données", ["real_or_fallback","real_only","fallback_only"], index=0,
    help="real_or_fallback = Yahoo sinon synthétique; real_only = Yahoo seulement; fallback_only = synthétique.")
st.session_state["data_mode"] = data_mode

# ===================== Tabs =====================
tab_scr, tab_watch, tab_bt, tab_live, tab_ai, tab_diag = st.tabs(["🔎 Screener","⭐ Watchlist","🧪 Backtest","📡 Live","🤖 AI Signals","🛠️ Diagnostics"])

# -------- Screener --------
with tab_scr:
    st.subheader("Screener (Actions / Crypto) — scoring multi-facteurs")
    capital = st.number_input("Capital (pour sizing indicatif)", value=100000.0, step=1000.0)
    risk_pct = st.number_input("Risque par trade (ex: 0.005 = 0.5%)", value=0.005, step=0.001, format="%.3f")
    mode = st.radio("Mode", ["Equities (actions)", "Crypto"], horizontal=True)

    if mode.startswith("Equities"):
        sc = cfg.get("screener_equities", {})
        symbols = st.multiselect("Tickers (actions)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","1h","30m","15m","5m"], index=0)
        lookback = st.slider("Lookback (jours)", 30, 365, sc.get("lookback_days", 120))
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10))
        weights = sc.get("score_weights", {"momentum":0.45,"volatility":0.2,"liquidity":0.2,"trend_quality":0.15})
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 2.5)))
        run = st.button("Lancer (actions)")
        if run and symbols:
            df = pd.DataFrame(build_screener_rows(symbols, interval, lookback, weights, atr_mult_target, is_crypto=False))
            df["rec_stop"], df["rec_size"] = 0.0, 0
            for i, sym in enumerate(df["symbol"]):
                data, _, _ = fetch(sym, interval, lookback, mode=data_mode)
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
        run = st.button("Lancer (crypto)", key="cr_run")
        if run and symbols:
            df = pd.DataFrame(build_screener_rows(symbols, interval, lookback, weights, atr_mult_target, is_crypto=True))
            df["rec_stop"], df["rec_size"] = 0.0, 0
            for i, sym in enumerate(df["symbol"]):
                data, _, _ = fetch(sym, interval, lookback, mode=data_mode)
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
    run_watch = st.button("Mettre à jour watchlist")
    if run_watch and wl_symbols:
        cards = []
        logs = {}
        for sym in wl_symbols:
            df, used_int, log = fetch(sym, wl_interval, wl_lookback, mode=st.session_state.get("data_mode","real_or_fallback"))
            logs[sym] = log
            if df.empty: 
                cards.append({"symbol": sym, "status": "vide"})
                continue
            c = s_close(df); price = nz(c, 0.0)
            a = atr(df, 14); atr_pct = float(nz(a,0.0) / price) if price>0 else 0.0
            e20 = ema(c,20); e50 = ema(c,50); e200 = ema(c,200); vw = vwap(df)
            ema_bull = (nz(e20,np.nan) > nz(e50,np.nan) > nz(e200,np.nan))
            breakout_20 = price >= float(c.tail(20).max())
            pullback_e20 = price >= nz(e20, price) and (price - nz(e20, price)) / max(price,1e-9) <= 0.02
            alerts = []
            if ema_bull: alerts.append("Tendance haussière EMA20>50>200")
            if breakout_20: alerts.append("Breakout 20-bar")
            if pullback_e20: alerts.append("Pullback sain vers EMA20")
            if atr_pct >= 0.04: alerts.append("Volatilité élevée (ATR%≥4%)")

            cards.append({
                "symbol": sym, "used_interval": used_int, "price": price,
                "ATR%": round(atr_pct*100,2),
                "EMA20": float(nz(e20, np.nan)), "EMA50": float(nz(e50, np.nan)), "EMA200": float(nz(e200, np.nan)),
                "VWAP": float(nz(vw, np.nan)),
                "Alerts": " • ".join(alerts) if alerts else "—"
            })
        if cards:
            dfw = pd.DataFrame(cards)
            st.dataframe(dfw, use_container_width=True)
            st.download_button("⬇️ Exporter watchlist (CSV)", data=dfw.to_csv(index=False).encode("utf-8"), file_name="watchlist.csv", mime="text/csv")
        st.session_state["yf_logs"] = logs

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
            df, _, _ = fetch(sym, interval, lookback_days, mode=st.session_state.get("data_mode","real_or_fallback"))
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)

            pnl = 0.0; trades = 0; wins = 0; losses = 0; rs = []
            h, m = map(int, end_liq.split(":")); end_t = dtime(hour=h, minute=m)

            for session, g in sig.groupby("session"):
                rows = g[g["entry"].notna()]
                if rows.empty: continue
                entry_ts = rows.index[0]
                direction = rows.iloc[0]["direction"]; entry = rows.iloc[0]["entry"]; stop = rows.iloc[0]["stop"]
                risk_per_share = abs(entry - stop); trades += 1; exited = False
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
        df, _, _ = fetch(live_symbol, interval, days, mode=st.session_state.get("data_mode","real_or_fallback"))
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

# -------- 🤖 AI Signals --------
with tab_ai:
    st.subheader("🤖 AI Signals — Prévision achat/vente (prochain bar)")
    aicfg = cfg.get("ai", {})
    ai_tickers = st.text_input("Tickers (séparés par des virgules)", ", ".join(aicfg.get("tickers", []))).strip()
    ai_symbols = [s.strip() for s in ai_tickers.split(",") if s.strip()] if ai_tickers else aicfg.get("tickers", [])
    ai_interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m","5m"], index=4)
    ai_lookback = st.slider("Lookback (jours)", 30, 365, aicfg.get("lookback_days", 90))
    epochs = st.number_input("Epochs (entraînement)", value=int(aicfg.get("train_epochs", 250)), min_value=50, max_value=1000, step=50)
    lr = st.number_input("Learning rate", value=float(aicfg.get("lr", 0.2)), step=0.05, format="%.2f")
    l2 = st.number_input("L2 régularisation", value=float(aicfg.get("l2", 1e-4)), step=1e-4, format="%.6f")
    buy_th = st.number_input("Seuil d'achat (P↑)", value=float(aicfg.get("buy_threshold", 0.6)), min_value=0.5, max_value=0.9, step=0.05)
    sell_th = st.number_input("Seuil de vente/short (P↑)", value=float(aicfg.get("sell_threshold", 0.4)), min_value=0.1, max_value=0.5, step=0.05)
    capital = st.number_input("Capital (pour sizing)", value=100000.0, step=1000.0, key="ai_cap")
    risk_pct = st.number_input("Risque par trade", value=0.005, step=0.001, format="%.3f", key="ai_risk")
    run_ai = st.button("🧠 Lancer la prédiction")

    if run_ai and ai_symbols:
        results = []
        for sym in ai_symbols:
            df, used_int, log = fetch(sym, ai_interval, ai_lookback, mode=st.session_state.get("data_mode","real_or_fallback"))
            if df is None or df.empty:
                results.append({"symbol": sym, "status": "vide"}); continue

            feats = build_features(df)
            if len(feats) < 60:
                # fallback rule-based
                p = rule_based_prob(df)
                c = s_close(df); a = atr(df,14)
                entry = nz(c, 0.0); stop = entry - 2.0 * nz(a,0.0)
                action = "BUY" if p >= buy_th else ("SELL" if p <= sell_th else "HOLD")
                size = int((capital * risk_pct) // max(abs(entry - stop), 1e-9)) if action != "HOLD" else 0
                results.append({"symbol": sym, "used_interval": used_int, "mode": "rule-based", "P_up": round(p*100,1),
                                "action": action, "entry": entry, "stop": stop, "size": size})
                continue

            # build X,y
            c = s_close(df).reindex(feats.index)
            y = (c.shift(-1) > c).astype(int).loc[feats.index]
            data = pd.concat([feats, y.rename("y")], axis=1).dropna()
            if len(data) < 60:
                p = rule_based_prob(df)
                entry = nz(c, 0.0); a = atr(df,14); stop = entry - 2.0 * nz(a,0.0)
                action = "BUY" if p >= buy_th else ("SELL" if p <= sell_th else "HOLD")
                size = int((capital * risk_pct) // max(abs(entry - stop), 1e-9)) if action != "HOLD" else 0
                results.append({"symbol": sym, "used_interval": used_int, "mode": "rule-based", "P_up": round(p*100,1),
                                "action": action, "entry": entry, "stop": stop, "size": size})
                continue

            X = data.drop(columns=["y"]).values.astype(float)
            yv = data["y"].values.astype(float)
            Xz, mu, sigma = standardize(X)
            # split train / keep last row for prediction
            X_train, y_train = Xz[:-1], yv[:-1]
            x_last = Xz[-1:]
            w, b = train_logreg_gd(X_train, y_train, lr=lr, epochs=epochs, l2=l2)
            p_up = float(predict_proba(x_last, w, b)[0])

            entry = float(c.loc[data.index[-1]])
            a = atr(df,14); atr_last = nz(a, 0.0)
            stop = entry - 2.0 * atr_last if p_up >= 0.5 else entry + 2.0 * atr_last
            action = "BUY" if p_up >= buy_th else ("SELL" if p_up <= sell_th else "HOLD")
            size = int((capital * risk_pct) // max(abs(entry - stop), 1e-9)) if action != "HOLD" else 0
            results.append({"symbol": sym, "used_interval": used_int, "mode": "logreg", "P_up": round(p_up*100,1),
                            "action": action, "entry": float(entry), "stop": float(stop), "size": int(size)})

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.caption("⚠️ Éducatif uniquement. Pas un conseil financier. Les marchés comportent des risques.")
        else:
            st.info("Aucun symbole valide.")

# -------- Diagnostics --------
with tab_diag:
    st.subheader("🛠️ Astuce & limite Yahoo")
    st.markdown(\"\"\"
- Sur des **intervalles fins** (5m, 15m, 30m…), Yahoo limite la période. L’app **baisse automatiquement l’intervalle** si besoin.
- Barre latérale → **Mode de données** :
  - `real_or_fallback` : essaye Yahoo, sinon **données synthétiques**
  - `real_only` : uniquement Yahoo (peut renvoyer vide)
  - `fallback_only` : uniquement synthétique (utile pour tester l’app hors-ligne)
- L’onglet **🤖 AI Signals** entraîne un petit modèle de régression logistique (numpy) sur tes **dernières données** pour estimer **P(up)** au prochain bar. 
  - Si l’historique est trop court, un **mode règles** calcule une proba simple basée sur EMA/RSI/MACD/Breakout.
  - Seuils par défaut : BUY si P↑ ≥ 0.60 ; SELL si P↑ ≤ 0.40 ; sinon HOLD.
- Toujours utiliser un money management strict. Ceci n’est **pas** un conseil d’investissement.
\"\"\")
