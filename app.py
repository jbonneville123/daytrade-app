# ---------- Live (simulation) ----------
with tab3:
    st.subheader("Live (simulation) — aperçu des signaux intraday")
    st.caption("Ce mode **ne place pas d'ordres**. C’est un aperçu éducatif.")
    live_symbol = st.text_input("Ticker", "AAPL")
    interval = st.selectbox("Intervalle (intraday)", ["5m","15m","30m"], index=0)
    days = st.slider("Jours à afficher", 1, 5, 1)
    ema_p = st.number_input("EMA period", value=20)
    atr_p = st.number_input("ATR period", value=14)
    atr_mult = st.number_input("ATR × stop", value=2.0)
    or_minutes = st.slider("Opening Range (min)", 5, 30, 15)
    allow_short = st.checkbox("Autoriser short", value=True)

    if st.button("Actualiser") and live_symbol:
        df = yf.download(live_symbol, period=f"{days}d", interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            st.warning("Pas de données.")
        else:
            # ⚠️ APLATIR ET NORMALISER
            df = flatten_cols(df)
            for c in ["Open","High","Low","Close","Adj Close","Volume"]:
                if c not in df.columns:
                    df[c] = np.nan
            df = df.rename(columns={"Adj Close": "AdjClose"})
            df["Close"], df["Volume"] = safe_close(df), safe_volume(df)

            sig = generate_signals(
                df,
                ema_period=ema_p,
                atr_period=atr_p,
                atr_mult=atr_mult,
                or_minutes=or_minutes,
                allow_short=allow_short
            )

            # ✅ PLOT SÉCURISÉ (évite KeyError MultiIndex)
            cols_to_plot = [c for c in ["Close", "EMA", "VWAP"] if c in sig.columns]
            if len(cols_to_plot) >= 2:
                to_plot = sig[cols_to_plot].copy()
                to_plot = flatten_cols(to_plot)   # au cas où
                if to_plot.dropna().shape[0] >= 5:
                    st.line_chart(to_plot)
                else:
                    st.info("Pas assez de points propres pour tracer. Essaie un autre intervalle ou plus de jours.")
            else:
                st.info("Colonnes manquantes pour le graphique (Close/EMA/VWAP).")

            # Tableau de sortie
            cols_last = [c for c in ["direction", "entry", "stop", "or_high", "or_low"] if c in sig.columns]
            if cols_last:
                st.table(sig.iloc[-1:][cols_last])
            else:
                st.info("Pas d’infos de signal disponibles pour la dernière ligne.")
