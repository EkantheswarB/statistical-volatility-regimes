import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def realized_volatility(returns: pd.Series, window: int = 5):
    """
    Compute rolling realized volatility over a given window.
    Formula: sqrt(sum of squared returns over 'window' days).
    """
    rv = (returns ** 2).rolling(window).sum() ** 0.5
    return rv

def evaluate_forecasts(
    forecasts_df_list,
    realized_vol_series,
    asset_name: str,
    results_dir: str = "../results",
    figs_dir: str = "../results/figures"
):
    """
    Compare model volatility forecasts with realized volatility.

    Parameters
    ----------
    forecasts_df_list : list of DataFrames
        Each contains columns [asset, model, last_date, one_day_ahead_vol_forecast_pct].
    realized_vol_series : pd.Series
        Rolling realized volatility (as fraction, not %).
    asset_name : str
        Asset label for plots and output filenames.
    results_dir : str
        Folder to save CSV outputs.
    figs_dir : str
        Folder to save figure outputs.

    Returns
    -------
    summary : pd.DataFrame
        Table of mean absolute error and RMSE by model.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Merge all forecast records into one table
    df_all = pd.concat(forecasts_df_list, ignore_index=True)

    # Take the most recent realized volatility value (converted to %)
    tail_rv = realized_vol_series.dropna().iloc[-1] * 100.0
    df_all["realized_vol_tail_pct"] = float(tail_rv)

    # Compute forecast errors
    df_all["abs_err"] = np.abs(
        df_all["one_day_ahead_vol_forecast_pct"] - df_all["realized_vol_tail_pct"]
    )
    df_all["sq_err"] = (
        df_all["one_day_ahead_vol_forecast_pct"] - df_all["realized_vol_tail_pct"]
    ) ** 2

    # Aggregate metrics
    summary = (
        df_all.groupby("model")
        .agg({"abs_err": "mean", "sq_err": "mean"})
    )
    summary["rmse"] = np.sqrt(summary["sq_err"])
    summary = summary.drop(columns=["sq_err"]).reset_index()

    # 📊 RMSE bar plot
    plt.figure()
    plt.bar(summary["model"], summary["rmse"], color="skyblue")
    plt.title(f"{asset_name}: Forecast RMSE vs Realized Volatility")
    plt.ylabel("RMSE (pct vol)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{asset_name}_forecast_rmse.png"))
    plt.close()

    # Save results as CSV
    out_csv = os.path.join(results_dir, f"{asset_name}_forecast_eval.csv")
    summary.to_csv(out_csv, index=False)

    return summary
