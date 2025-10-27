import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def fit_hmm_regimes(
    returns: pd.Series,
    n_states: int = 2,
    asset_name: str = "asset",
    results_dir: str = "../results",
    figs_dir: str = "../results/figures"
):
    """
    Fit a Gaussian Hidden Markov Model (HMM) to infer low-vol/high-vol regimes.

    Parameters
    ----------
    returns : pd.Series
        Log returns of the asset.
    n_states : int
        Number of latent states (default = 2).
    asset_name : str
        Name of the asset (for labeling plots).
    results_dir : str
        Directory to save results (CSV, etc.).
    figs_dir : str
        Directory to save figures.

    Returns
    -------
    dict with keys:
        'model' : fitted HMM model object
        'posterior_probs' : DataFrame of state probabilities
        'regime_series' : DataFrame with most likely state per date
        'trans_mat' : transition probability matrix
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Create feature matrix (returns + squared returns)
    r = returns.dropna().to_frame(name="ret")
    r["ret_sq"] = r["ret"] ** 2

    scaler = StandardScaler()
    X = scaler.fit_transform(r[["ret", "ret_sq"]])

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )
    hmm.fit(X)

    hidden_states = hmm.predict(X)
    posterior_probs = hmm.predict_proba(X)

    # 1️⃣ Posterior probability plot
    plt.figure(figsize=(10, 4))
    for state in range(n_states):
        plt.plot(r.index, posterior_probs[:, state], label=f"State {state}")
    plt.title(f"{asset_name}: Regime posterior probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{asset_name}_regime_probs.png"))
    plt.close()

    # 2️⃣ Returns scatter colored by regime
    plt.figure(figsize=(10, 4))
    for state in range(n_states):
        mask = hidden_states == state
        plt.scatter(
            r.index[mask],
            r["ret"][mask] * 100,
            s=6,
            alpha=0.6,
            label=f"State {state}"
        )
    plt.title(f"{asset_name}: Returns colored by inferred regime")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{asset_name}_regime_scatter.png"))
    plt.close()

    # 3️⃣ Transition matrix heatmap
    trans_mat = hmm.transmat_
    plt.figure()
    plt.imshow(trans_mat, cmap="Blues")
    for i in range(n_states):
        for j in range(n_states):
            plt.text(j, i, f"{trans_mat[i, j]:.2f}", ha="center", va="center")
    plt.title(f"{asset_name}: HMM transition matrix")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.colorbar(label="P(transition)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{asset_name}_transition_matrix.png"))
    plt.close()

    # Prepare output
    regime_df = pd.DataFrame({
        "date": r.index,
        "hidden_state": hidden_states
    }).set_index("date")

    return {
        "model": hmm,
        "posterior_probs": pd.DataFrame(
            posterior_probs,
            index=r.index,
            columns=[f"state_{i}" for i in range(n_states)]
        ),
        "regime_series": regime_df,
        "trans_mat": trans_mat
    }
