[README.md](https://github.com/user-attachments/files/22007003/README.md)
# DayTrade App (zéro code)

Cette application Streamlit te permet de:
- **Screener** actions et crypto (classement + gain potentiel estimé)
- **Backtest** simple de la stratégie ORB + EMA20 + VWAP + stop ATR
- **Live (simulation)** : aperçu des signaux intraday (pas d’ordres réels)

## Démarrage rapide (Windows / Mac / Linux)
1) Installe Python 3.10+
2) Ouvre un terminal dans le dossier et exécute:
```
pip install -r requirements.txt
```
3) Lance l’app:
```
streamlit run app.py
```
OU double-clique **Start-Windows.bat** (Windows) / **Start-Mac.command** (Mac).

## Déploiement simple (en ligne)
- Mets ce dossier sur GitHub puis déploie sur **Streamlit Cloud**
- Dans les *Secrets* (si un jour tu ajoutes Alpaca), ajoute `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`.

### Notes importantes
- Contenu **éducatif** uniquement. Utiliser d’abord en **simulation**.
- Les données intraday gratuites de `yfinance` sont limitées. Pour du pro, prévoir un data provider payant.
