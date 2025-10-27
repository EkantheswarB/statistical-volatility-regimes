import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

def fit_garch_models(
    returns: pd.Series,
    asset_name: str,
    results_dir: str = "../results",
    figs_dir: str = "../results/figures"
):
    """
    Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) models to a return series.

    Outputs:
        - Conditional volatility plots
        - QQ-plots with Ljung-Box p-value annotation
        - 1-step-ahead volatility forecast

    Returns:
        models_dict: dict of fitted model result objects
        forecasts_df: DataFrame with one-day-ahead volatility forecasts
        cond_vol_dict: {model_name: Series of conditional volatility (% terms)}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Drop NaN and scale to percent to stabilize fitting
    r = returns.dropna() * 100.0

    models_dict = {}

    # 1️⃣ GARCH(1,1)
    garch11 = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
    garch11_res = garch11.fit(disp="off")
    models_dict["GARCH11"] = garch11_res

    # 2️⃣ EGARCH(1,1)
    egarch = arch_model(r, vol="EGARCH", p=1, q=1, mean="Constant", dist="normal")
    egarch_res = egarch.fit(disp="off")
    models_dict["EGARCH"] = egarch_res

    # 3️⃣ GJR-GARCH(1,1)
    gjr = arch_model(r, vol="GARCH", p=1, o=1, q=1, mean="Constant", dist="normal")
    gjr_res = gjr.fit(disp="off")
    models_dict["GJRGARCH"] = gjr_res

    forecasts_records = []
    cond_vol_dict = {}

    for model_name, res in models_dict.items():
        cond_vol = res.conditional_volatility  # pct volatility
        cond_vol_dict[model_name] = cond_vol

        # One-step-ahead forecast
        fcast = res.forecast(horizon=1, reindex=False)
        var_fcast = fcast.variance.values[-1, 0]
        vol_fcast = np.sqrt(var_fcast)

        forecasts_records.append({
            "asset": asset_name,
            "model": model_name,
            "last_date": cond_vol.index[-1],
            "one_day_ahead_vol_forecast_pct": float(vol_fcast),
        })

        # 📈 Plot conditional volatility
        plt.figure()
        plt.plot(cond_vol.index, cond_vol, label=f"{model_name} cond. vol")
        plt.title(f"{asset_name}: {model_name} conditional volatility")
        plt.xlabel("Date")
        plt.ylabel("Vol (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"{asset_name}_{model_name}_volatility.png"))
        plt.close()

        # 📊 Residual diagnostics: QQ-plot + Ljung-Box test
        std_resid = res.std_resid.dropna()
        lb_stat, lb_p = acorr_ljungbox(std_resid, lags=[10], return_df=False)

        plt.figure()
        stats.probplot(std_resid, dist="norm", plot=plt)
        plt.title(
            f"{asset_name}: {model_name} standardized residual QQ-plot\n"
            f"Ljung-Box p(10)={lb_p[0]:.3f}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"{asset_name}_{model_name}_qqplot.png"))
        plt.close()

    forecasts_df = pd.DataFrame(forecasts_records)
    return models_dict, forecasts_df, cond_vol_dict
