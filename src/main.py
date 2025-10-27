"""
main.py
Author: Ekantheswar Bandarupalli
Project: Statistical Modeling of Volatility and Regime Switching in Financial Markets
Subtitle: Volatility Clustering and Hidden Regime Dynamics in SPX and BTC

Usage:
1. pip install -r requirements.txt
2. python src/main.py

Pipeline:
- Downloads SPY & BTC data from Yahoo Finance (2015–today)
- Computes log returns
- Fits GARCH, EGARCH, GJR-GARCH models
- Fits Hidden Markov Model (HMM) for regime detection
- Plots conditional vs realized vol & regime heatmaps
- Evaluates 1-day-ahead volatility forecasts
"""

import os
import pandas as pd
from data_loader import load_data
from garch_model import fit_garch_models
from regime_switching import fit_hmm_regimes
from diagnostics import realized_volatility, evaluate_forecasts
from visualization import plot_conditional_vol_vs_realized

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figures")

def main():
    # 1️⃣ Load and prepare data
    spy_ret, btc_ret, both = load_data(output_dir=DATA_DIR)

    # 2️⃣ Fit GARCH-family models
    spy_models, spy_fc, spy_cond_vol = fit_garch_models(spy_ret["log_ret"], "SPY", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)
    btc_models, btc_fc, btc_cond_vol = fit_garch_models(btc_ret["log_ret"], "BTC", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)

    # 3️⃣ Realized volatility (5-day rolling)
    spy_rv = realized_volatility(spy_ret["log_ret"])
    btc_rv = realized_volatility(btc_ret["log_ret"])

    # 4️⃣ Plot conditional vs realized volatility
    plot_conditional_vol_vs_realized("SPY", spy_cond_vol, spy_rv, figs_dir=FIGS_DIR)
    plot_conditional_vol_vs_realized("BTC", btc_cond_vol, btc_rv, figs_dir=FIGS_DIR)

    # 5️⃣ Regime switching (Hidden Markov Model)
    spy_hmm = fit_hmm_regimes(spy_ret["log_ret"], n_states=2, asset_name="SPY", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)
    btc_hmm = fit_hmm_regimes(btc_ret["log_ret"], n_states=2, asset_name="BTC", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)

    # 6️⃣ Evaluate volatility forecast accuracy
    spy_eval = evaluate_forecasts([spy_fc], spy_rv, "SPY", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)
    btc_eval = evaluate_forecasts([btc_fc], btc_rv, "BTC", results_dir=RESULTS_DIR, figs_dir=FIGS_DIR)

    # 7️⃣ Save combined summary
    forecasts_path = os.path.join(RESULTS_DIR, "forecasts.csv")
    pd.concat([
        spy_eval.assign(asset="SPY"),
        btc_eval.assign(asset="BTC")
    ]).to_csv(forecasts_path, index=False)

    print("✅ Pipeline complete.")
    print(f"Figures saved to: {FIGS_DIR}")
    print(f"Forecast metrics saved to: {forecasts_path}")

if __name__ == "__main__":
    main()
