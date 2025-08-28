# app.py — DayTrade App (fixes: config robuste, MultiIndex, scores NaN, charts sûrs)
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App", layout="wide")
st.title("📊 DayTrade App — Screener, Backtest, Live (simulation)")

# ===================== Config robuste =====================
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
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f) or {}
            return _merge_cfg(cfg, DEFAULT_CFG)
        return DEFAULT_CFG
    except Exception:
        return DEFAULT_CFG

cfg = load_config()
if not os.path.exists("config.yaml"):
    st.warning("⚠️ `config.yaml` introuvable — configuration par défaut utilisée.")

# ===================== Helpers sûrs (MultiIndex/NaN) =====================
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Aplati un éventuel MultiIndex de colonnes (yfinance)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Retourne toujours une Series 1D float pour la colonne demandée."""
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        # si MultiIndex -> prend la 1ère sous-colonne
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")

def s_open(df):   return safe_column(df, "Open")
def s_high(df):   return safe_column(df, "High")
def s_low(df):    return safe_column(df, "Low")
def s_close(df):  return safe_column(df, "Close")
def s_volume(df): return safe_column(df, "Volume")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = s_high(df), s_low(df), s_close(df)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    close, vol = s_close(df), s_volume(df)
    pv = (close * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return pv / vv

def finite(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)

def nz(series: pd.Series, default=0.0):
    """Dernière valeur finie d'une Series, sinon default."""
    if series is None or len(series) == 0:
        return float(default)
    return finite(series.iloc[-1], default)

# ===================== Téléchargement data (1 symbole) =====================
def fetch(symbol: str, period_days: int = 200, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(tickers=symbol, period=f"{period_days}d", interval=interval, auto_adjust=False, progress=False, prepost=False)
    if df.empty:
        return df
    df = flatten_cols(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["Open","High","Low","Close","AdjClose","Volume"]].copy()

# ===================== ORB + EMA + VWAP + Stop ATR =====================
def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = pd.to_datetime(df.index)
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def first_true_index(cond) -> pd.Timestamp | None:
    """Renvoie le premier index True (accepte Series/DataFrame)."""
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
    # colonnes sûres
    df["Open"], df["High"], df["Low"] = s_open(df), s_high(df), s_low(df)
    df["Close"], df["Volume"] = s_close(df), s_volume(df)
    # indicateurs
    df["EMA"] = ema(df["Close"], ema_period)
    df["ATR"] = atr(df, atr_period)
    df["VWAP"] = vwap(df)
    df["session"] = df.index.normalize()
    # sorties
    for c in ["direction","entry","stop","or_high","or_low"]:
        df[c] = None

    for session, g in df.groupby("session"):
        mask_or = compute_opening_range_mask(g, minutes=or_minutes)
        if mask_or.sum() == 0:
            continue
        or_high = finite(g.loc[mask_or, "High"].max(), np.nan)
        or_low  = finite(g.loc[mask_or, "Low"].min(),  np.nan)
        after = g.loc[~mask_or].copy()
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

        entry   = finite(after.at[entry_idx, "Close"], np.nan)
        atr_val = finite(after.at[entry_idx, "ATR"],   np.nan)
        if not np.isfinite(entry) or not np.isfinite(atr_val):
            continue
        stop = max(or_low, entry - atr_mult * atr_val) if direction == "long" else min(or_high, entry + atr_mult * atr_val)

        ix = g.index[g.index >= entry_idx]
        df.loc[ix, ["direction","entry","stop","or_high","or_low"]] = [direction, entry, stop, or_high, or_low]

    return df

def safe_avg_dollar_volume(df: pd.DataFrame, window: int = 30) -> float:
    if df is None or df.empty:
        return 0.0
    c = pd.to_numeric(s_close(df).tail(window), errors="coerce")
    v = pd.to_numeric(s_volume(df).tail(window), errors="coerce")
    prod = (c * v).dropna()
    return float(prod.mean()) if not prod.empty else 0.0

# ===================== UI =====================
tab1, tab2, tab3 = st.tabs(["🔎 Screener", "🧪 Backtest", "📡 Live (simulation)"])

# ---------- Screener ----------
with tab1:
    st.subheader("Screener séparé (Actions / Crypto)")
    mode = st.radio("Mode", ["Equities (actions)", "Crypto"], horizontal=True)

    # ---- Actions ----
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
                price_last = nz(s_close(df), 0.0)
                avg_vol = finite(s_volume(df).tail(30).mean(), 0.0)
                if price_last < min_price or avg_vol < min_avg_vol:
                    continue

                close = s_close(df)
                n = len(close)
                if n < 6 or close.isna().all():
                    continue

                # features robustes
                if n >= 22 and pd.notna(close.iloc[-21]) and close.iloc[-21] != 0:
                    mom = finite((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21], 0.0)
                elif n >= 6 and pd.notna(close.iloc[-5]) and close.iloc[-5] != 0:
                    mom = finite((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5], 0.0)
                else:
                    mom = 0.0

                atr_s = atr(df, 14)
                atr_last = nz(atr_s, 0.0)
                atr_pct = finite(atr_last / price_last if price_last > 0 else 0.0, 0.0)
                avg_dvol = finite(safe_avg_dollar_volume(df, 30), 0.0)

                e20 = nz(ema(close,20), np.nan)
                e50 = nz(ema(close,50), np.nan) if n >= 50 else np.nan
                e200 = nz(ema(close,200), np.nan) if n >= 200 else np.nan
                if np.isfinite(e20) and np.isfinite(e50) and np.isfinite(e200):
                    tq = 1.0 if (e20>e50>e200) else (0.5 if (e20>e50 or e50>e200) else 0.0)
                elif np.isfinite(e20) and np.isfinite(e50):
                    tq = 0.7 if e20 > e50 else 0.3
                else:
                    tq = 0.5

                potential = finite(
                    atr_mult_target * atr_last if potential_method=="atr_target"
                    else max(0.0, float(close.tail(60).max()) - price_last),
                0.0)

                # normalisations bornées
                mom_n = float(np.clip(mom, -0.8, 0.8))
                vol_n = float(np.clip(atr_pct, 0.0, 0.5))
                liq_n = float(np.log10(max(avg_dvol, 1.0))/8.0)

                score = finite(
                    weights['momentum']*mom_n +
                    weights['volatility']*vol_n +
                    weights['liquidity']*liq_n +
                    weights['trend_quality']*tq,
                0.0)

                rows.append({
                    "symbol": sym, "price": price_last, "score": score, "momentum_21d": mom, "atr_pct": atr_pct,
                    "avg_dollar_volume": avg_dvol, "trend_quality": tq, "potential_gain_$": potential,
                    "ema20": e20, "ema50": e50, "ema200": e200
                })

            if rows:
                out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(topn))
                st.dataframe(out, use_container_width=True)
            else:
                st.info("Aucun symbole ne passe les filtres.")

    # ---- Crypto (tolérant aux volumes manquants) ----
    else:
        sc = cfg.get("screener_crypto", {})
        symbols = st.multiselect("Tickers (crypto)", sc.get("tickers", []), default=sc.get("tickers", []))
        interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m","5m"], index=0)
        lookback = st.slider("Lookback (jours)", 30, 365, max(30, sc.get("lookback_days", 120)))
        topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10))
        weights = sc.get("score_weights", {'momentum':0.55,'volatility':0.25,'liquidity':0.15,'trend_quality':0.05})
        potential_method = st.selectbox("Méthode de gain potentiel", ["atr_target","recent_high"], index=0)
        atr_mult_target = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 3.0)))
        ignore_liquidity = st.checkbox("Ignorer la liquidité si Volume=0", value=True)
        force_results = st.checkbox("Forcer un Top N (tolérant)", value=True)
        run = st.button("Lancer le screener (crypto)")

        if run and symbols:
            rows = []
            for sym in symbols:
                df = fetch(sym, period_days=max(lookback+20, 80), interval=interval)
                if df.empty:
                    if force_results:
                        rows.append({"symbol": sym, "price": 0.0, "score": -1e9,
                                     "momentum_21d": 0.0, "atr_pct": 0.0, "avg_dollar_volume": 0.0,
                                     "trend_quality": 0.0, "potential_gain_$": 0.0,
                                     "ema20": np.nan, "ema50": np.nan, "ema200": np.nan})
                    continue

                close = s_close(df)
                n = len(close)
                price_last = nz(close, 0.0)

                # momentum adaptatif
                if n >= 22 and pd.notna(close.iloc[-21]) and close.iloc[-21] != 0:
                    mom = finite((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21], 0.0)
                elif n >= 6 and pd.notna(close.iloc[-5]) and close.iloc[-5] != 0:
                    mom = finite((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5], 0.0)
                else:
                    mom = 0.0

                atr_s = atr(df, 14)
                atr_last = nz(atr_s, 0.0)
                atr_pct = finite(atr_last / price_last if price_last > 0 else 0.0, 0.0)

                # liquidité: proxy via ATR% si volume inutilisable
                vol_last30 = s_volume(df).tail(30)
                if not vol_last30.isna().all() and vol_last30.sum() > 0 and price_last > 0:
                    avg_dvol = finite((close.tail(30) * vol_last30).dropna().mean(), 0.0)
                    liq_n = float(np.log10(max(avg_dvol, 1.0)) / 8.0)
                else:
                    avg_dvol = 0.0
                    liq_n = float(np.clip(atr_pct * 2.0, 0.0, 1.0)) if ignore_liquidity else 0.0

                # EMAs & tendance
                e20  = nz(ema(close, 20), np.nan) if n >= 1 else np.nan
                e50  = nz(ema(close, 50), np.nan) if n >= 50 else np.nan
                e200 = nz(ema(close,200), np.nan) if n >= 200 else np.nan
                if np.isfinite(e20) and np.isfinite(e50) and np.isfinite(e200):
                    tq = 1.0 if (e20 > e50 > e200) else (0.5 if (e20 > e50 or e50 > e200) else 0.0)
                elif np.isfinite(e20) and np.isfinite(e50):
                    tq = 0.7 if e20 > e50 else 0.3
                else:
                    tq = 0.5

                # potentiel
                if potential_method == "atr_target":
                    potential = finite(atr_mult_target * atr_last, 0.0)
                else:
                    look = close.tail(min(60, n))
                    recent_high = finite(look.max() if len(look) else price_last, price_last)
                    potential = finite(max(0.0, recent_high - price_last), 0.0)

                # normalisations
                mom_n = float(np.clip(mom, -1.0, 1.0))
                vol_n = float(np.clip(atr_pct, 0.0, 0.8))

                score = finite(
                    weights['momentum']      * mom_n +
                    weights['volatility']    * vol_n +
                    weights['liquidity']     * liq_n +
                    weights['trend_quality'] * tq,
                0.0)

                if force_results or price_last > 0:
                    rows.append({
                        "symbol": sym, "price": price_last, "score": score, "momentum_21d": mom,
                        "atr_pct": atr_pct, "avg_dollar_volume": avg_dvol, "trend_quality": tq,
                        "potential_gain_$": potential, "ema20": e20, "ema50": e50, "ema200": e200
                    })

            if rows:
                out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(topn))
                st.dataframe(out, use_container_width=True)
            else:
                st.info("Aucune crypto renvoyée — ajoute des tickers, augmente le lookback, ou active 'Forcer un Top N'.")

# ---------- Backtest ----------
with tab2:
    st.subheader("Backtest ORB + EMA20 + VWAP + Stop ATR (simplifié)")
    bcfg = cfg.get("backtest", {})
    tickers = st.multiselect("Tickers à backtester", bcfg.get("tickers", []), default=bcfg.get("tickers", []))
    interval = st.selectbox("Intervalle", ["5m","15m","30m","1h"], index=0, key="bt_int")
    lookback_days = st.slider("Historique (jours)", 5, 30, bcfg.get("lookback_days", 10), key="bt_days")
    or_minutes = st.slider("Opening Range (minutes)", 5, 30, bcfg.get("or_minutes", 15), key="bt_or")
    ema_p = st.number_input("EMA period", value=int(bcfg.get("ema_period", 20)), key="bt_ema")
    atr_p = st.number_input("ATR period", value=int(bcfg.get("atr_period", 14)), key="bt_atr")
    atr_mult = st.number_input("ATR × (stop)", value=float(bcfg.get("atr_mult", 2.0)), key="bt_mult")
    allow_short = st.checkbox("Autoriser short", value=bool(bcfg.get("allow_short", True)), key="bt_short")
    end_liq = st.text_input("Liquidation (HH:MM)", bcfg.get("end_liquidate", "15:55"), key="bt_liq")
    capital = st.number_input("Capital de départ", value=float(bcfg.get("starting_capital", 100000)), key="bt_cap")
    risk_per_trade = st.number_input("Risque par trade (ex: 0.005 = 0.5%)", value=float(bcfg.get("risk_per_trade", 0.005)), key="bt_rpt")
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
            df = fetch(sym, period_days=lookback_days, interval=interval)
            if df.empty:
                continue
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
        df = fetch(live_symbol, period_days=days, interval=interval)
        if df.empty:
            st.warning("Pas de données.")
        else:
            sig = generate_signals(df, ema_period=ema_p, atr_period=atr_p, atr_mult=atr_mult, or_minutes=or_minutes, allow_short=allow_short)

            # Graphique sécurisé
            cols_to_plot = [c for c in ["Close","EMA","VWAP"] if c in sig.columns]
            if len(cols_to_plot) >= 2 and sig[cols_to_plot].dropna().shape[0] >= 5:
                st.line_chart(sig[cols_to_plot], use_container_width=True)
            else:
                st.info("Pas assez de colonnes/points propres pour tracer (essaie plus de jours/un autre intervalle).")

            last_cols = [c for c in ["direction","entry","stop","or_high","or_low"] if c in sig.columns]
            if last_cols:
                st.table(sig.iloc[-1:][last_cols])
            else:
                st.info("Pas d’infos de signal disponibles pour la dernière ligne.")
