# Statistical Modeling of Volatility and Regime Switching in Financial Markets  
### Volatility Clustering and Hidden Regime Dynamics in SPX and BTC  

**Author:** Ekantheswar Bandarupalli  
**Date:** October 2025  
**Keywords:** Volatility Modeling, GARCH, Hidden Markov Models, Regime Switching, SPX, BTC, Financial Econometrics  

---

## 🚀 Overview
This project models volatility dynamics and hidden market regimes in both traditional equities (SPY) and crypto (BTC-USD).  
It combines:
- GARCH-family conditional volatility modeling (GARCH, EGARCH, GJR-GARCH)
- Hidden Markov Models (HMM) for regime switching
- Forecast evaluation of 1-day-ahead volatility

---

## 🧠 Motivation
Volatility is not constant — markets shift between calm and turbulent states.  
BTC, in particular, demonstrates persistent high-volatility regimes, unlike SPY’s fast-reverting volatility cycles.

This repo offers reproducible research that highlights:
- Volatility clustering patterns  
- Hidden regime persistence  
- Model-based forecast diagnostics  

---

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python src/main.py
