# app.py — DayTrade App (version "symbiose")
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import time

st.set_page_config(page_title="DayTrade App", layout="wide")
st.title("📊 DayTrade App — Screener, Backtest, Live (symbiose)")

# ========= Helpers communs (une seule source de vérité) =========
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Si yfinance renvoie des colonnes MultiIndex, on garde uniquement le dernier niveau."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Retourne une Series 1D float pour la colonne demandée (robuste MultiIndex/DataFrame)."""
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        try:
            col = col.iloc[:, 0]
        except Exception:
            col = col.squeeze()
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

def fetch_single(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    """
    Télécharge UN seul symbole (évite MultiIndex par ticker),
    aplatit, normalise les colonnes, renvoie un DataFrame standard.
    """
    df = yf.download(
        tickers=symbol,
        period=f"{period_days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False
    )
    if df.empty:
        return df
    df = flatten_cols(df)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.rename(columns={"Adj Close": "AdjClose"})
    # ordre de colonnes stable
    df = df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]].copy()
    # index datetime
    df.index = pd.to_datetime(df.index)
    return df

def compute_opening_range_mask(df: pd.DataFrame, minutes: int = 15) -> pd.Series:
    idx = pd.to_datetime(df.index)
    or_start = idx.normalize() + pd.Timedelta(hours=9, minutes=30)
    or_end = or_start + pd.Timedelta(minutes=minutes)
    return (idx >= or_start) & (idx < or_end)

def first_true_index(cond) -> pd.Timestamp | None:
    """Renvoie le premier index où la condition est True (accepte Series/DataFrame)."""
    if isinstance(cond, pd.DataFrame):
        cond = cond.any(axis=1)
    if isinstance(cond, pd.Series):
        cond = cond.fillna(False)
        idx = cond[cond].index
        return idx[0] if len(idx) else None
    return None

def generate_signals(df: pd.DataFrame, ema_period=20, atr_period=14, atr_mult=2.0, or_minutes=15, allow_short=True) -> pd.DataFrame:
    """
    Stratégie ORB + EMA + VWAP avec stop ATR — robustifiée pour Live/Screener/Backtest.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = flatten_cols(df)

    # Sanitize colonnes
    df["Open"], df["High"], df["Low"] = s_open(df), s_high(df), s_low(df)
    df["Close"], df["Volume"] = s_close(df), s_volume(df)

    # Indicateurs
    df["EMA"] = ema(df["Close"], ema_period)
    df["ATR"] = atr(df, atr_period)
    df["VWAP"] = vwap(df)

    df.index = pd.to_datetime(df.index)
    df["session"] = df.index.normalize()

    # Sorties
    df["direction"] = None
    df["entry"] = None
    df["stop"] = None
    df["or_high"] = None
    df["or_low"] = None

    # Boucle par séance
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

        e_long = first_true_index(long_cond)
        e_short = first_true_index(short_cond) if allow_short else None

        entry_idx = None
        direction = None
        if e_long is not None and (e_short is None or e_long <= e_short):
            entry_idx, direction = e_long, "long"
        elif e_short is not None:
            entry_idx, direction = e_short, "short"
        if entry_idx is None:
            continue

        entry = float(after.loc[entry_idx, "Close"])
        atr_val = float(after.loc[entry_idx, "ATR"])
        stop = max(or_low, entry - atr_mult*atr_val) if direction == "long" else min(or_high, entry + atr_mult*atr_val)

        ix = g.index[g.index >= entry_idx]
        df.loc[ix, "direction"] = direction
        df.loc[ix, "entry"] = entry
        df.loc[ix, "stop"] = stop
        df.loc[ix, "or_high"] = or_high
        df.loc[ix, "or_low"] = or_low

    return df

def safe_avg_dollar_volume(df: pd.DataFrame, window: int = 30) -> float:
    if df is None or df.empty:
        return 0.0
    c = s_close(df).tail(window)
    v = s_volume(df).tail(window)
    prod = (c * v).dropna()
    return float(prod.mean()) if not prod.empty else 0.0

# ========= Config =========
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ========= UI : 3 onglets (clés uniques par onglet) =========
tab_scr, tab_bt, tab_live = st.tabs(["🔎 Screener", "🧪 Backtest", "📡 Live"])

# ------------------ Screener ------------------
with tab_scr:
    st.subheader("Screener séparé (Actions / Crypto)")
    scr_mode = st.radio("Mode", ["Equities (actions)", "Crypto"], horizontal=True, key="scr_mode")

    if scr_mode.startswith("Equities"):
        sc = cfg.get("screener_equities", {})
        scr_tickers = st.multiselect("Tickers (actions)", sc.get("tickers", []), default=sc.get("tickers", []), key="scr_eq_tickers")
        scr_interval = st.selectbox("Intervalle", ["1d","1h","30m","15m","5m"], index=0, key="scr_eq_interval")
        scr_lookback = st.slider("Lookback (jours)", 60, 365, sc.get("lookback_days", 120), key="scr_eq_lookback")
        scr_min_price = st.number_input("Prix minimum", value=float(sc.get("min_price", 3)), key="scr_eq_minprice")
        scr_min_avgvol = st.number_input("Volume moyen min (actions)", value=float(sc.get("min_avg_volume", 1_000_000)), key="scr_eq_minavgvol")
        scr_topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10), key="scr_eq_topn")
        scr_weights = sc.get("score_weights", {'momentum':0.5,'volatility':0.2,'liquidity':0.2,'trend_quality':0.1})
        scr_pot_method = st.selectbox("Méthode gain potentiel", ["atr_target","recent_high"], index=0, key="scr_eq_potm")
        scr_atr_mult_tgt = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 2.5)), key="scr_eq_atr_tgt")
        if st.button("Lancer le screener (actions)", key="scr_eq_run") and scr_tickers:
            rows = []
            for sym in scr_tickers:
                df = fetch_single(sym, period_days=max(scr_lookback+80, 220), interval=scr_interval)
                if df.empty:
                    continue
                close_series = s_close(df)
                price = float(close_series.tail(1).fillna(0).iloc[-1]) if len(close_series) else 0.0
                avg_vol = float(s_volume(df).tail(30).mean() or 0.0)
                if price < scr_min_price or avg_vol < scr_min_avgvol:
                    continue
                if len(close_series) < 60 or close_series.isna().all():
                    continue
                # scoring
                mom = float((close_series.iloc[-1] - close_series.iloc[-21]) / close_series.iloc[-21]) if pd.notna(close_series.iloc[-21]) and close_series.iloc[-21] != 0 else 0.0
                atr_s = atr(df, 14)
                atr_pct = float((atr_s.iloc[-1] / close_series.iloc[-1])) if pd.notna(close_series.iloc[-1]) and close_series.iloc[-1] != 0 else 0.0
                avg_dvol = safe_avg_dollar_volume(df, 30)
                e20, e50, e200 = ema(close_series,20).iloc[-1], ema(close_series,50).iloc[-1], ema(close_series,200).iloc[-1]
                tq = 1.0 if (e20>e50>e200) else (0.5 if (e20>e50 or e50>e200) else 0.0)
                potential = float(scr_atr_mult_tgt * atr_s.iloc[-1]) if scr_pot_method=="atr_target" else float(max(0.0, close_series.iloc[-60:].max() - close_series.iloc[-1]))
                mom_n = float(np.clip(mom, -0.5, 0.5))
                vol_n = float(np.clip(atr_pct, 0.0, 0.25))
                liq_n = float(np.log10(max(avg_dvol, 1.0))/8.0)
                score = scr_weights['momentum']*mom_n + scr_weights['volatility']*vol_n + scr_weights['liquidity']*liq_n + scr_weights['trend_quality']*tq
                rows.append({
                    "symbol": sym, "price": price, "score": score, "momentum_21d": mom, "atr_pct": atr_pct,
                    "avg_dollar_volume": avg_dvol, "trend_quality": tq, "potential_gain_$": potential,
                    "ema20": e20, "ema50": e50, "ema200": e200
                })
            if rows:
                out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(scr_topn))
                st.dataframe(out, use_container_width=True)
            else:
                st.info("Aucun symbole ne passe les filtres.")

   # ---- Screener Crypto (TOUJOURS retourne un Top N) ----
else:
    sc = cfg.get("screener_crypto", {})
    scrc_tickers = st.multiselect("Tickers (crypto)", sc.get("tickers", []), default=sc.get("tickers", []), key="scr_cr_tickers")
    scrc_interval = st.selectbox("Intervalle", ["1d","4h","1h","30m","15m","5m"], index=0, key="scr_cr_interval")
    scrc_lookback = st.slider("Lookback (jours)", 30, 365, max(30, sc.get("lookback_days", 120)), key="scr_cr_lookback")
    scrc_topn = st.number_input("Top N", 1, 50, sc.get("top_n", 10), key="scr_cr_topn")
    scrc_weights = sc.get("score_weights", {'momentum':0.55,'volatility':0.25,'liquidity':0.15,'trend_quality':0.05})
    scrc_pot_method = st.selectbox("Méthode gain potentiel", ["atr_target","recent_high"], index=0, key="scr_cr_potm")
    scrc_atr_mult_tgt = st.number_input("ATR × (objectif)", value=float(sc.get("atr_mult_target", 3.0)), key="scr_cr_atr_tgt")

    # Options de tolérance
    scrc_ignore_liquidity = st.checkbox("Ignorer la liquidité (si Volume=0 sur yfinance)", value=True, key="scr_cr_ignoreliq")
    scrc_force_results = st.checkbox("Forcer un Top N (désactiver la plupart des filtres)", value=True, key="scr_cr_force")

    if st.button("Lancer le screener (crypto)", key="scr_cr_run") and scrc_tickers:
        rows = []
        for sym in scrc_tickers:
            df = fetch_single(sym, period_days=max(scrc_lookback+20, 80), interval=scrc_interval)
            if df.empty:
                # En mode forcé, on garde quand même une ligne placeholder (score très faible)
                if scrc_force_results:
                    rows.append({"symbol": sym, "price": 0.0, "score": -1e9, "momentum_21d": 0.0,
                                 "atr_pct": 0.0, "avg_dollar_volume": 0.0, "trend_quality": 0.0,
                                 "potential_gain_$": 0.0, "ema20": np.nan, "ema50": np.nan, "ema200": np.nan})
                continue

            close = s_close(df)
            price_last = float(close.iloc[-1]) if len(close) and pd.notna(close.iloc[-1]) else 0.0

            # Fenêtres adaptatives: on prend ce qu’on a
            n = len(close)
            mom_window = 21 if n > 21 else (n-1 if n >= 5 else 0)
            if mom_window > 0 and close.iloc[-mom_window] not in (0, np.nan):
                mom = float((close.iloc[-1] - close.iloc[-mom_window]) / close.iloc[-mom_window])
            else:
                mom = 0.0

            # ATR et volatilité %
            atr_s = atr(df, 14)
            atr_last = float(atr_s.iloc[-1]) if len(atr_s) and pd.notna(atr_s.iloc[-1]) else 0.0
            atr_pct = float(atr_last / price_last) if price_last > 0 else 0.0

            # Liquidité: si Volume inutilisable, proxy via volatilité
            vol_last30 = s_volume(df).tail(30)
            if not vol_last30.isna().all() and vol_last30.sum() > 0 and price_last > 0:
                avg_dvol = float((close.tail(30) * vol_last30).dropna().mean() or 0.0)
                liq_n = float(np.log10(max(avg_dvol, 1.0)) / 8.0)
            else:
                avg_dvol = 0.0
                liq_n = float(np.clip(atr_pct * 2.0, 0.0, 1.0)) if scrc_ignore_liquidity else 0.0

            # EMAs et trend quality
            e20  = float(ema(close, 20).iloc[-1])  if n >= 20 else np.nan
            e50  = float(ema(close, 50).iloc[-1])  if n >= 50 else np.nan
            e200 = float(ema(close, 200).iloc[-1]) if n >= 200 else np.nan
            if not np.isnan(e20) and not np.isnan(e50) and not np.isnan(e200):
                tq = 1.0 if (e20 > e50 > e200) else (0.5 if (e20 > e50 or e50 > e200) else 0.0)
            elif not np.isnan(e20) and not np.isnan(e50):
                tq = 0.7 if e20 > e50 else 0.3
            else:
                tq = 0.5  # défaut neutre en mode forcé

            # Gain potentiel
            if scrc_pot_method == "atr_target":
                potential = float(scrc_atr_mult_tgt * atr_last)
            else:
                look = close.iloc[-min(60, n):]
                recent_high = float(look.max()) if len(look) else price_last
                potential = float(max(0.0, recent_high - price_last))

            # Normalisations robustes
            mom_n = float(np.clip(mom, -1.0, 1.0))
            vol_n = float(np.clip(atr_pct, 0.0, 0.8))

            score = (
                scrc_weights['momentum']      * mom_n +
                scrc_weights['volatility']    * vol_n +
                scrc_weights['liquidity']     * liq_n +
                scrc_weights['trend_quality'] * tq
            )

            # En mode strict OFF (forcé), on n’exclut personne ; en mode strict ON, on peut filtrer un peu
            if scrc_force_results:
                rows.append({
                    "symbol": sym, "price": price_last, "score": score, "momentum_21d": mom,
                    "atr_pct": atr_pct, "avg_dollar_volume": avg_dvol, "trend_quality": tq,
                    "potential_gain_$": potential, "ema20": e20, "ema50": e50, "ema200": e200
                })
            else:
                # filtres doux (ex: prix positif)
                if price_last > 0:
                    rows.append({
                        "symbol": sym, "price": price_last, "score": score, "momentum_21d": mom,
                        "atr_pct": atr_pct, "avg_dollar_volume": avg_dvol, "trend_quality": tq,
                        "potential_gain_$": potential, "ema20": e20, "ema50": e50, "ema200": e200
                    })

        if rows:
            out = pd.DataFrame(rows).sort_values("score", ascending=False).head(int(scrc_topn))
            st.dataframe(out, use_container_width=True)
        else:
            st.info("Aucune crypto renvoyée — ajoute des tickers, augmente le lookback, ou active 'Forcer un Top N'.")
    
# ------------------ Live ------------------
with tab_live:
    st.subheader("Live (simulation) — aperçu des signaux intraday")
    st.caption("Ce mode **ne place pas d'ordres**. C’est un aperçu éducatif.")
    live_symbol = st.text_input("Ticker", "AAPL", key="live_symbol")
    live_interval = st.selectbox("Intervalle (intraday)", ["5m","15m","30m"], index=0, key="live_interval")
    live_days = st.slider("Jours à afficher", 1, 5, 1, key="live_days")
    live_ema_p = st.number_input("EMA period", value=20, key="live_ema")
    live_atr_p = st.number_input("ATR period", value=14, key="live_atr")
    live_atr_mult = st.number_input("ATR × stop", value=2.0, key="live_atr_mult")
    live_or_minutes = st.slider("Opening Range (min)", 5, 30, 15, key="live_or")
    live_allow_short = st.checkbox("Autoriser short", value=True, key="live_short")

    if st.button("Actualiser", key="live_run") and live_symbol:
        df = fetch_single(live_symbol, period_days=live_days, interval=live_interval)
        if df.empty:
            st.warning("Pas de données.")
        else:
            sig = generate_signals(
                df,
                ema_period=live_ema_p,
                atr_period=live_atr_p,
                atr_mult=live_atr_mult,
                or_minutes=live_or_minutes,
                allow_short=live_allow_short
            )

            # Plot sécurisé
            cols_to_plot = [c for c in ["Close", "EMA", "VWAP"] if c in sig.columns]
            if len(cols_to_plot) >= 2:
                to_plot = sig[cols_to_plot].copy()
                to_plot = flatten_cols(to_plot)
                if to_plot.dropna().shape[0] >= 5:
                    st.line_chart(to_plot, use_container_width=True)
                else:
                    st.info("Pas assez de points propres pour tracer. Essaie un autre intervalle ou plus de jours.")
            else:
                st.info("Colonnes manquantes pour le graphique (Close/EMA/VWAP).")

            cols_last = [c for c in ["direction", "entry", "stop", "or_high", "or_low"] if c in sig.columns]
            if cols_last:
                st.table(sig.iloc[-1:][cols_last])
            else:
                st.info("Pas d’infos de signal pour la dernière ligne.")
