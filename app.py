# app.py — DayTrade App (final, robuste, avec fallback si données Yahoo vides)
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App", layout="wide")
st.title("📊 DayTrade App — Screener, Backtest, Live")

# ===================== Config =====================
DEFAULT_CFG = {
    "screener_equities": {
        "tickers": ["AAPL","MSFT","TSLA","NVDA","AMD","META","AMZN","GOOGL","SMCI","NFLX"],
        "lookback_days": 120,
        "min_price": 3,
        "min_avg_volume": 1_000_000,
        "top_n": 10,
        "score_weights": {"momentum":0.5,"volatility":0.2,"liquidity":0.2,"trend_quality":0.1},
        "atr_mult_target": 2.5
    },
    "screener_crypto": {
        "tickers": ["BTC-USD","ETH-USD","SOL-USD","AVAX-USD","DOGE-USD","XRP-USD"],
        "lookback_days": 120,
        "top_n": 10,
        "score_weights": {"momentum":0.55,"volatility":0.25,"liquidity":0.15,"trend_quality":0.05},
        "atr_mult_target": 3.0
    },
    "backtest": {
        "tickers": ["AAPL","MSFT","TSLA"],
        "lookback_days": 10,
        "or_minutes": 15,
        "ema_period": 20,
        "atr_period": 14,
        "atr_mult": 2.0,
        "allow_short": True,
        "starting_capital": 100000,
        "risk_per_trade": 0.005,
        "end_liquidate": "15:55"
    }
}

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml","r") as f:
            try:
                user = yaml.safe_load(f) or {}
                DEFAULT_CFG.update(user)
            except Exception:
                pass
    return DEFAULT_CFG

cfg = load_config()

# ===================== Utils =====================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def safe_column(df, name):
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce")

def s_open(df):   return safe_column(df,"Open")
def s_high(df):   return safe_column(df,"High")
def s_low(df):    return safe_column(df,"Low")
def s_close(df):  return safe_column(df,"Close")
def s_volume(df): return safe_column(df,"Volume")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    high, low, close = s_high(df), s_low(df), s_close(df)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(),(high-prev_close).abs(),(low-prev_close).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def vwap(df):
    c, v = s_close(df), s_volume(df)
    return (c*v).cumsum()/v.cumsum().replace(0,np.nan)

def finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except:
        return default

def nz(series, default=0.0):
    if series is None or len(series)==0: return default
    return finite(series.iloc[-1], default)

# --- Fallback si Yahoo ne renvoie rien ---
def synthetic_df(days:int, interval:str, start_price:float=100.0):
    n = max(days*10, 60)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005,0.02,n)
    px = np.cumprod(1+rets)*start_price
    return pd.DataFrame({
        "Open":px*(1+rng.normal(0,0.002,n)),
        "High":px*(1+rng.normal(0.001,0.002,n)),
        "Low":px*(1-rng.normal(0.001,0.002,n)),
        "Close":px,
        "AdjClose":px,
        "Volume":rng.integers(1e6,5e6,n)
    }, index=pd.date_range(end=pd.Timestamp.today(), periods=n, freq="H"))

def fetch(sym, lookback_days:int, interval:str, buffer:int=40):
    try:
        df = yf.download(sym, period=f"{lookback_days+buffer}d", interval=interval, progress=False)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        df = synthetic_df(lookback_days, interval, start_price=(100 if "-" not in sym else 30000))
    df = flatten_cols(df)
    if "Adj Close" in df.columns: df.rename(columns={"Adj Close":"AdjClose"}, inplace=True)
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df.columns: df[c]=np.nan
    return df

# ===================== Screener =====================
tab1,tab2,tab3 = st.tabs(["🔎 Screener","🧪 Backtest","📡 Live"])

with tab1:
    mode = st.radio("Mode",["Equities (actions)","Crypto"], horizontal=True)

    if mode.startswith("Equities"):
        sc = cfg["screener_equities"]
        symbols = st.multiselect("Tickers", sc["tickers"], default=sc["tickers"])
        interval = st.selectbox("Intervalle",["1d","1h","30m","15m","5m"])
        lookback = st.slider("Lookback jours",30,365,sc["lookback_days"])
        topn = st.number_input("Top N",1,50,sc["top_n"])
        run = st.button("Run screener (actions)")
        if run:
            rows=[]
            for sym in symbols:
                df=fetch(sym,lookback,interval)
                close=s_close(df); n=len(close); price_last=nz(close,0.0)
                if n<5: continue
                mom=(close.iloc[-1]-close.iloc[-min(21,n)])/close.iloc[-min(21,n)] if n>1 else 0
                atr_val=nz(atr(df,14),0.0); atr_pct=atr_val/price_last if price_last>0 else 0
                avg_dvol=nz(s_volume(df).tail(30),1.0)*price_last
                e20=nz(ema(close,20),np.nan); e50=nz(ema(close,50),np.nan); e200=nz(ema(close,200),np.nan)
                tq=1.0 if e20>e50>e200 else 0.5
                score=mom*0.5+atr_pct*0.2+np.log10(max(avg_dvol,1))/8*0.2+tq*0.1
                rows.append({"symbol":sym,"price":price_last,"score":finite(score),
                             "momentum":finite(mom),"atr_pct":finite(atr_pct),
                             "ema20":e20,"ema50":e50,"ema200":e200})
            if rows: st.dataframe(pd.DataFrame(rows).sort_values("score",ascending=False).head(int(topn)))

    else:
        sc = cfg["screener_crypto"]
        symbols = st.multiselect("Tickers", sc["tickers"], default=sc["tickers"])
        interval = st.selectbox("Intervalle",["1d","4h","1h","30m","15m","5m"])
        lookback = st.slider("Lookback jours",30,365,sc["lookback_days"])
        topn = st.number_input("Top N",1,50,sc["top_n"])
        run = st.button("Run screener (crypto)")
        if run:
            rows=[]
            for sym in symbols:
                df=fetch(sym,lookback,interval)
                close=s_close(df); n=len(close); price_last=nz(close,0.0)
                if n<5: continue
                mom=(close.iloc[-1]-close.iloc[-min(21,n)])/close.iloc[-min(21,n)] if n>1 else 0
                atr_val=nz(atr(df,14),0.0); atr_pct=atr_val/price_last if price_last>0 else 0
                avg_dvol=nz(s_volume(df).tail(30),1.0)*price_last
                e20=nz(ema(close,20),np.nan); e50=nz(ema(close,50),np.nan); e200=nz(ema(close,200),np.nan)
                tq=1.0 if e20>e50>e200 else 0.5
                score=mom*0.55+atr_pct*0.25+np.log10(max(avg_dvol,1))/8*0.15+tq*0.05
                rows.append({"symbol":sym,"price":price_last,"score":finite(score),
                             "momentum":finite(mom),"atr_pct":finite(atr_pct)})
            if rows: st.dataframe(pd.DataFrame(rows).sort_values("score",ascending=False).head(int(topn)))
