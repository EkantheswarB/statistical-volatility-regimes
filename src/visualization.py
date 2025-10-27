import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_conditional_vol_vs_realized(
    asset_name: str,
    cond_vol_dict,
    realized_vol_series: pd.Series,
    figs_dir: str = "../results/figures"
):
    """
    Compare conditional volatility from GARCH-family models to realized volatility.
    Saves a combined line plot for visual comparison.
    """
    os.makedirs(figs_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    for model_name, vol_series in cond_vol_dict.items():
        plt.plot(vol_series.index, vol_series, label=f"{model_name} cond vol")
    plt.plot(
        realized_vol_series.index,
        realized_vol_series * 100,
        label="realized vol",
        linestyle="--",
        linewidth=1
    )
    plt.title(f"{asset_name}: Conditional vs Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Vol (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{asset_name}_cond_vs_realized.png"))
    plt.close()
