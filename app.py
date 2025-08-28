
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App", layout="wide")
st.title("📊 DayTrade App — Screener, Backtest, Live (simulation)")

# ---------- Helpers (indicateurs & data) ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["Close"] * df["Volume"]).cumsum()
    vv = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vv

def fetch(symbol: str, period_days: int = 200, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(tickers=symbol, period=f"{period_days}d", interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["Open","High","Low","Close","AdjClose","Volume"]].copy()

def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = df.index
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def generate_signals(df: pd.DataFrame, ema_period=20, atr_period=14, atr_mult=2.0, or_minutes=15, allow_short=True):
    if df.empty:
        return df
    df = df.copy()
    df["EMA"] = ema(df["Close"], ema_period)
    df["ATR"] = atr(df, atr_period)
    df["VWAP"] = vwap(df)
    df["session"] = df.index.normalize()
    df["direction"] = None
    df["entry"] = None
    df["stop"] = None
    df["or_high"] = None
    df["or_low"] = None

    for session, g in df.groupby("session"):
        mask_or = compute_opening_range_mask(g, minutes=or_minutes)
        if mask_or.sum() == 0:
            continue
        or_high = g.loc[mask_or, "High"].max()
        or_low = g.loc[mask_or, "Low"].min()
        after = g.loc[~mask_or].copy()
        if after.empty:
            continue

        long_cond = (after["Close"] > or_high) & (after["EMA"] > after["VWAP"]) & (after["Volume"] > 0)
        short_cond = (after["Close"] < or_low) & (after["EMA"] < after["VWAP"]) & (after["Volume"] > 0)

        entry_idx = None
        direction = None
        if long_cond.any():
            entry_idx = long_cond.idxmax()
            direction = "long"
        if allow_short and short_cond.any():
            s_idx = short_cond.idxmax()
            if entry_idx is None or s_idx < entry_idx:
                entry_idx = s_idx
                direction = "short"

        if entry_idx is None:
            continue

        entry = after.at[entry_idx, "Close"]
        atr_val = after.at[entry_idx, "ATR"]
        stop = max(or_low, entry - atr_mult * atr_val) if direction == "long" else min(or_high, entry + atr_mult * atr_val)
        ix = g.index[g.index >= entry_idx]
        df.loc[ix, "direction"] = direction
        df.loc[ix, "entry"] = entry
        df.loc[ix, "stop"] = stop
        df.loc[ix, "or_high"] = or_high
        df.loc[ix, "or_low"] = or_low

    return df

# ---------- Load config ----------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

tab1, tab2, tab3 = st.tabs(["🔎 Screener", "🧪 Backtest", "📡 Live (simulation)"])

# ---------- Screener ----------
with tab1:
    st.subheader("Screener séparé (Actions / Crypto)")
    mode = st.radio("Mode", ["Equities (actions)", "Crypto"], horizontal=True)

    if mode.startswith("Equities"):
        sc = cfg.get("screener_equities", {})
        symbols = st.multiselect("Tickers (actions)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","1h","30m","15m","5m"], index=0)
        lookback = st.slider("Lookback (jours)", 60, 365, sc.get("lookback_days", 120))
        min_price = st.number_input("Prix minimum", value=float(sc.get("min_price", 3)))
        min_avg_vol = st.number_input("Volume moyen min (actions)", value=float(sc.get("min_avg_volume", 1_000_000)))
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10))
        weights = sc.get("score_weights", {'momentum':0.5,'volatility':0.2,'liquidity':0.2,'trend_quality':0.1})
        potential_method = st.selectbox("Méthode de gain potentiel", ["atr_target","recent_high"], index=0)
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 2.5)))
        run = st.button("Lancer le screener (actions)")

        if run and symbols:
            rows = []
            for sym in symbols:
                df = fetch(sym, period_days=max(lookback+80, 220), interval=interval)
                if df.empty:
                    continue
                price = float(df["Close"].iloc[-1])
                avg_vol = float(df["Volume"].iloc[-30:].mean()) if not df["Volume"].iloc[-30:].isna().all() else 0.0
                if price < min_price: 
                    continue
                if avg_vol < min_avg_vol:
                    continue
                # scoring
                close = df["Close"]
                if len(close) < 60:
                    continue
                mom = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if close.iloc[-21] != 0 else 0.0
                atr_s = atr(df, 14)
                atr_pct = (atr_s.iloc[-1] / close.iloc[-1]) if close.iloc[-1] != 0 else 0.0
                avg_dvol = (close.iloc[-30:] * df["Volume"].iloc[-30:]).mean()
                e20,e50,e200 = ema(close,20).iloc[-1], ema(close,50).iloc[-1], ema(close,200).iloc[-1]
                tq = 1.0 if (e20>e50>e200) else (0.5 if (e20>e50 or e50>e200) else 0.0)
                potential = atr_mult_target * atr_s.iloc[-1] if potential_method=="atr_target" else max(0.0, close.iloc[-60:].max()-close.iloc[-1])

                mom_n = float(np.clip(mom, -0.5, 0.5))
                vol_n = float(np.clip(atr_pct, 0.0, 0.25))
                liq_n = float(np.log10(max(avg_dvol, 1.0))/8.0)
                score = weights['momentum']*mom_n + weights['volatility']*vol_n + weights['liquidity']*liq_n + weights['trend_quality']*tq

                rows.append({
                    "symbol": sym, "price": price, "score": score, "momentum_21d": mom, "atr_pct": atr_pct,
                    "avg_dollar_volume": avg_dvol, "trend_quality": tq, "potential_gain_$": potential,
                    "ema20": e20, "ema50": e50, "ema200": e200
                })
            if rows:
                out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(topn))
                st.dataframe(out, use_container_width=True)
            else:
                st.info("Aucun symbole ne passe les filtres.")

    else:
        sc = cfg.get("screener_crypto", {})
        symbols = st.multiselect("Tickers (crypto)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m","5m"], index=0)
        lookback = st.slider("Lookback (jours)", 60, 365, sc.get("lookback_days", 120))
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10))
        weights = sc.get("score_weights", {'momentum':0.55,'volatility':0.25,'liquidity':0.15,'trend_quality':0.05})
        potential_method = st.selectbox("Méthode de gain potentiel", ["atr_target","recent_high"], index=0)
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 3.0)))
        run = st.button("Lancer le screener (crypto)")

        if run and symbols:
            rows = []
            for sym in symbols:
                df = fetch(sym, period_days=max(lookback+80, 220), interval=interval)
                if df.empty: 
                    continue
                close = df["Close"]
                if len(close) < 60:
                    continue
                mom = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if close.iloc[-21] != 0 else 0.0
                atr_s = atr(df, 14)
                atr_pct = (atr_s.iloc[-1] / close.iloc[-1]) if close.iloc[-1] != 0 else 0.0
                avg_dvol = (close.iloc[-30:] * df["Volume"].iloc[-30:]).mean()
                e20,e50,e200 = ema(close,20).iloc[-1], ema(close,50).iloc[-1], ema(close,200).iloc[-1]
                tq = 1.0 if (e20>e50>e200) else (0.5 if (e20>e50 or e50>e200) else 0.0)
                potential = atr_mult_target * atr_s.iloc[-1] if potential_method=="atr_target" else max(0.0, close.iloc[-60:].max()-close.iloc[-1])

                mom_n = float(np.clip(mom, -0.5, 0.5))
                vol_n = float(np.clip(atr_pct, 0.0, 0.35))
                liq_n = float(np.log10(max(avg_dvol, 1.0))/8.0)
                score = weights['momentum']*mom_n + weights['volatility']*vol_n + weights['liquidity']*liq_n + weights['trend_quality']*tq

                rows.append({
                    "symbol": sym, "price": float(close.iloc[-1]), "score": score, "momentum_21d": mom, "atr_pct": atr_pct,
                    "avg_dollar_volume": avg_dvol, "trend_quality": tq, "potential_gain_$": potential,
                    "ema20": e20, "ema50": e50, "ema200": e200
                })
            if rows:
                out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(topn))
                st.dataframe(out, use_container_width=True)
            else:
                st.info("Aucune crypto ne passe les filtres.")

# ---------- Backtest ----------
with tab2:
    st.subheader("Backtest ORB + EMA20 + VWAP + Stop ATR (simplifié)")
    bcfg = cfg.get("backtest", {})
    tickers = st.multiselect("Tickers à backtester", bcfg.get("tickers", []), default=bcfg.get("tickers", []))
    interval = st.selectbox("Intervalle", ["5m","15m","30m","1h"], index=0)
    lookback_days = st.slider("Historique (jours)", 5, 30, bcfg.get("lookback_days", 10))
    or_minutes = st.slider("Opening Range (minutes)", 5, 30, bcfg.get("or_minutes", 15))
    ema_p = st.number_input("EMA period", value=int(bcfg.get("ema_period", 20)))
    atr_p = st.number_input("ATR period", value=int(bcfg.get("atr_period", 14)))
    atr_mult = st.number_input("ATR × (stop)", value=float(bcfg.get("atr_mult", 2.0)))
    allow_short = st.checkbox("Autoriser short", value=bool(bcfg.get("allow_short", True)))
    end_liq = st.text_input("Liquidation (HH:MM)", bcfg.get("end_liquidate", "15:55"))
    capital = st.number_input("Capital de départ", value=float(bcfg.get("starting_capital", 100000)))
    risk_per_trade = st.number_input("Risque par trade (ex: 0.005 = 0.5%)", value=float(bcfg.get("risk_per_trade", 0.005)))
    runbt = st.button("Lancer le backtest")

    def position_size(cap, entry, stop, rpt):
        if entry is None or stop is None or entry <= 0:
            return 0
        rps = abs(entry - stop)
        if rps <= 0:
            return 0
        return int((cap * rpt) // rps)

    if runbt and tickers:
        results = []
        for sym in tickers:
            df = yf.download(tickers=sym, period=f"{lookback_days}d", interval=interval, prepost=False, auto_adjust=False, progress=False)
            if df.empty: 
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close":"AdjClose"})
            for c in ["Open","High","Low","Close","AdjClose","Volume"]:
                if c not in df.columns: df[c] = np.nan
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)

            pnl = 0.0
            trades = 0
            h, m = map(int, end_liq.split(":"))
            end_t = time(hour=h, minute=m)

            for session, g in sig.groupby("session"):
                rows = g[g["entry"].notna()]
                if rows.empty:
                    continue
                entry_ts = rows.index[0]
                direction = rows.iloc[0]["direction"]
                entry = rows.iloc[0]["entry"]
                stop = rows.iloc[0]["stop"]
                qty = position_size(capital, entry, stop, risk_per_trade)
                if qty <= 0:
                    continue
                trades += 1
                exited = False
                for ts, row in g.loc[g.index >= entry_ts].iterrows():
                    if ts.time() >= end_t:
                        price = row["Close"]
                        pnl += (price - entry) * qty if direction == "long" else (entry - price) * qty
                        exited = True
                        break
                    if direction == "long":
                        if row["Low"] <= stop:
                            pnl += (stop - entry) * qty
                            exited = True
                            break
                    else:
                        if row["High"] >= stop:
                            pnl += (entry - stop) * qty
                            exited = True
                            break
                    a = row["ATR"]
                    stop = max(stop, row["Close"] - atr_mult * a) if direction == "long" else min(stop, row["Close"] + atr_mult * a)
                if not exited:
                    last = g.iloc[-1]["Close"]
                    pnl += (last - entry) * qty if direction == "long" else (entry - last) * qty

            results.append({"symbol": sym, "trades": trades, "pnl": float(pnl)})
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("pnl", ascending=False), use_container_width=True)
        else:
            st.info("Pas de résultats sur cette période.")

# ---------- Live (simulation) ----------
with tab3:
    st.subheader("Live (simulation) — aperçu des signaux intraday")
    st.caption("Ce mode **ne place pas d'ordres**. C’est un aperçu éducatif.")
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
        df = yf.download(tickers=live_symbol, period=f"{days}d", interval=interval, prepost=False, auto_adjust=False, progress=False)
        if df.empty:
            st.warning("Pas de données.")
        else:
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "AdjClose"})
            for c in ["Open","High","Low","Close","AdjClose","Volume"]:
                if c not in df.columns: df[c] = np.nan
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)
            st.line_chart(sig[["Close","EMA","VWAP"]])
            st.table(sig.iloc[-1:][["direction","entry","stop","or_high","or_low"]])
