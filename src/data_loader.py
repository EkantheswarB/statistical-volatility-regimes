import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def download_asset(ticker: str, start="2015-01-01", end=None):
    """
    Downloads historical daily price data for the given ticker from Yahoo Finance.
    Returns a DataFrame with 'close' prices only.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def compute_log_returns(price_df: pd.DataFrame):
    """
    Computes log returns: log(P_t / P_{t-1})
    """
    ret = np.log(price_df["close"]).diff().dropna()
    return ret.to_frame(name="log_ret")

def load_data(output_dir: str = "../data"):
    """
    Downloads SPY and BTC-USD data, computes log returns,
    aligns them by date, and saves as CSV files.
    Returns:
        spy_ret, btc_ret, both
    """
    os.makedirs(output_dir, exist_ok=True)

    spy_raw = download_asset("SPY")
    btc_raw = download_asset("BTC-USD")

    spy_ret = compute_log_returns(spy_raw)
    btc_ret = compute_log_returns(btc_raw)

    both = (
        spy_ret.rename(columns={"log_ret": "spy_ret"})
        .join(btc_ret.rename(columns={"log_ret": "btc_ret"}), how="inner")
    )

    spy_ret.to_csv(os.path.join(output_dir, "spx.csv"), index_label="date")
    btc_ret.to_csv(os.path.join(output_dir, "btc.csv"), index_label="date")
    both.to_csv(os.path.join(output_dir, "aligned_returns.csv"), index_label="date")

    return spy_ret, btc_ret, both
