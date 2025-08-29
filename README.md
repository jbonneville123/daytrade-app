# DayTrade App Pro — All-in-One (YF Strong + AI Signals)

Ajouts :
- Onglet **🤖 AI Signals** : modèle **logistic regression (numpy)** entraîné à la volée sur tes données récentes pour prédire **P(up) au prochain bar**. 
- Pas de dépendances ML externes (zéro scikit-learn).

Toujours inclus :
- Strong Yahoo fetch (retries, stratégies multiples, intervalle auto-downgrade, fallback synthétique)
- Screener Equities/Crypto (scoring multi-facteurs + sizing)
- ⭐ Watchlist (RSI/EMA/VWAP/ATR% + alerts)
- Backtest ORB + EMA + VWAP + ATR stop
- Live (simulation)

## Lancer
```
pip install -r requirements.txt
streamlit run app.py
```
