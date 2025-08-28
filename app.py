import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App", layout="wide")
st.title("📊 DayTrade App — Screener, Backtest, Live (simulation)")

# ---------- Helpers ----------
def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        try:
            col = col.iloc[:, 0]
        except Exception:
            col = col.squeeze()
    return pd.to_numeric(col, errors="coerce")

def safe_open(df): return safe_column(df, "Open")
def safe_high(df): return safe_column(df, "High")
def safe_low(df): return safe_column(df, "Low")
def safe_close(df): return safe_column(df, "Close")
def safe_volume(df): return safe_column(df, "Volume")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = safe_high(df), safe_low(df), safe_close(df)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    close, vol = safe_close(df), safe_volume(df)
    pv = (close * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return pv / vv

def fetch(symbol: str, period_days: int = 200, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False, auto_adjust=False)
    if df.empty: return df
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c not in df.columns: df[c] = np.nan
    df = df.rename(columns={"Adj Close":"AdjClose"})
    return df[["Open","High","Low","Close","AdjClose","Volume"]]

def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = pd.to_datetime(df.index)
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def first_true_index(cond) -> pd.Timestamp | None:
    if isinstance(cond, pd.DataFrame):
        cond = cond.any(axis=1)
    if isinstance(cond, pd.Series):
        idx = cond.fillna(False)
        return idx[idx].index[0] if idx.any() else None
    return None

def generate_signals(df, ema_period=20, atr_period=14, atr_mult=2.0, or_minutes=15, allow_short=True):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df["Open"], df["High"], df["Low"] = safe_open(df), safe_high(df), safe_low(df)
    df["Close"], df["Volume"] = safe_close(df), safe_volume(df)
    df["EMA"], df["ATR"], df["VWAP"] = ema(df["Close"], ema_period), atr(df, atr_period), vwap(df)
    df.index = pd.to_datetime(df.index)
    df["session"] = df.index.normalize()
    df["direction"], df["entry"], df["stop"] = None, None, None

    for session, g in df.groupby("session"):
        mask_or = compute_opening_range_mask(g, minutes=or_minutes)
        if mask_or.sum() == 0: continue
        or_high, or_low = g.loc[mask_or,"High"].max(), g.loc[mask_or,"Low"].min()
        after = g.loc[~mask_or]
        if after.empty: continue

        long_cond = (after["Close"] > or_high) & (after["EMA"] > after["VWAP"]) & (after["Volume"] > 0)
        short_cond = (after["Close"] < or_low) & (after["EMA"] < after["VWAP"]) & (after["Volume"] > 0)

        e_long, e_short = first_true_index(long_cond), first_true_index(short_cond) if allow_short else None
        entry_idx, direction = None, None
        if e_long and (not e_short or e_long <= e_short):
            entry_idx, direction = e_long, "long"
        elif e_short:
            entry_idx, direction = e_short, "short"
        if not entry_idx: continue

        entry, atr_val = after.loc[entry_idx,"Close"], after.loc[entry_idx,"ATR"]
        stop = max(or_low, entry - atr_mult*atr_val) if direction=="long" else min(or_high, entry + atr_mult*atr_val)
        df.loc[df.index>=entry_idx, ["direction","entry","stop"]] = [direction, entry, stop]

    return df

# ---------- Load config ----------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

tab1, tab2, tab3 = st.tabs(["🔎 Screener", "🧪 Backtest", "📡 Live (simulation)"])

# ---------- Live (simulation) ----------
with tab3:
    st.subheader("Live (simulation) — aperçu des signaux intraday")
    live_symbol = st.text_input("Ticker", "AAPL")
    interval = st.selectbox("Intervalle", ["5m","15m","30m"], index=0)
    days = st.slider("Jours à afficher", 1, 5, 1)
    ema_p = st.number_input("EMA period", 20)
    atr_p = st.number_input("ATR period", 14)
    atr_mult = st.number_input("ATR × stop", 2.0)
    or_minutes = st.slider("Opening Range (min)", 5, 30, 15)
    allow_short = st.checkbox("Autoriser short", True)
    if st.button("Actualiser") and live_symbol:
        df = yf.download(live_symbol, period=f"{days}d", interval=interval, progress=False)
        if df.empty:
            st.warning("Pas de données.")
        else:
            for c in ["Open","High","Low","Close","Adj Close","Volume"]:
                if c not in df.columns: df[c] = np.nan
            df = df.rename(columns={"Adj Close":"AdjClose"})
            df["Close"], df["Volume"] = safe_close(df), safe_volume(df)
            sig = generate_signals(df, ema_p, atr_p, atr_mult, or_minutes, allow_short)
            if sig[["Close","EMA","VWAP"]].dropna().shape[0] >= 5:
                st.line_chart(sig[["Close","EMA","VWAP"]])
            st.table(sig.iloc[-1:][["direction","entry","stop"]])
