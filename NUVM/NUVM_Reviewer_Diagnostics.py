"""
Reviewer diagnostics for the NUVM project.

This file addresses the boundary estimate a_hat = 0.999 by adding:

1. Likelihood profile for a.
2. Method-of-moments estimator.
3. Simulated recovery study for a = 0.2, 0.5, 0.8.
4. Multiple real-data applications: NIFTY, S&P 500, NASDAQ, Gold, Crude Oil.
5. Publication-style figures:
   - histogram with fitted densities
   - QQ plots
   - likelihood profile for a
   - RMSE versus n

Examples:

    python NUVM_Reviewer_Diagnostics.py --part profile
    python NUVM_Reviewer_Diagnostics.py --part mom
    python NUVM_Reviewer_Diagnostics.py --part recovery --replications 1000
    python NUVM_Reviewer_Diagnostics.py --part multi
    python NUVM_Reviewer_Diagnostics.py --part figures
    python NUVM_Reviewer_Diagnostics.py --part all
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


OUTPUT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = OUTPUT_DIR / "NUVM_Simulation_and_Real_Data.py"
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_base_module():
    spec = importlib.util.spec_from_file_location("nuvm_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nuvm = load_base_module()


def method_of_moments_nuvm(data):
    """
    Method-of-moments estimator for NUVM.

    For NUVM:
        E[X] = mu
        Var(X) = sigma^2
        kurtosis = 3 + a^2

    Hence:
        mu_hat = sample mean
        sigma_hat = sample standard deviation
        a_hat = sqrt(max(sample kurtosis - 3, 0))

    Because the model requires 0 < a < 1, a_hat is clipped to [1e-4, 0.999].
    A value clipped at 0.999 means the sample kurtosis is beyond the NUVM
    fourth-moment range.
    """
    data = np.asarray(data, dtype=float)
    mu_hat = float(np.mean(data))
    centered = data - mu_hat
    variance = float(np.mean(centered**2))
    sigma_hat = float(np.sqrt(max(variance, 1e-12)))
    sample_kurtosis = float(np.mean(centered**4) / max(variance**2, 1e-12))
    raw_a = float(np.sqrt(max(sample_kurtosis - 3.0, 0.0)))
    a_hat = float(np.clip(raw_a, 1e-4, 0.999))
    return {
        "mu_mom": mu_hat,
        "sigma_mom": sigma_hat,
        "a_mom_raw": raw_a,
        "a_mom": a_hat,
        "sample_kurtosis": sample_kurtosis,
        "boundary_flag": raw_a >= 0.999,
    }


def profile_loglik_a(data, a_grid=None, maxiter=1500):
    """
    Profile log-likelihood for a.

    For each fixed a, optimize only mu and sigma:
        l_p(a) = max_{mu, sigma} l(mu, sigma, a)
    """
    data = np.asarray(data, dtype=float)
    if a_grid is None:
        a_grid = np.r_[np.arange(0.05, 1.0, 0.05), [0.975, 0.99, 0.995, 0.999]]
        a_grid = np.unique(np.round(a_grid, 4))

    data_scale = max(float(np.std(data, ddof=0)), 1e-4)
    mu_start = float(np.mean(data))
    sigma_start = data_scale
    bounds = [
        (float(np.min(data) - 5.0 * data_scale), float(np.max(data) + 5.0 * data_scale)),
        (1e-6, 10.0 * data_scale),
    ]

    rows = []
    previous = np.array([mu_start, sigma_start], dtype=float)

    for a_value in a_grid:
        def objective(theta):
            mu_value, sigma_value = theta
            loglik = nuvm.nuvm_loglik((mu_value, sigma_value, float(a_value)), data)
            if not np.isfinite(loglik):
                return 1e100
            return -loglik

        result = minimize(
            objective,
            previous,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
        )
        if np.isfinite(result.fun):
            previous = result.x
        rows.append(
            {
                "a": float(a_value),
                "mu_hat_given_a": float(result.x[0]),
                "sigma_hat_given_a": float(result.x[1]),
                "loglik": float(-result.fun),
                "success": bool(np.isfinite(result.fun)),
                "message": str(result.message),
            }
        )

    profile = pd.DataFrame(rows)
    max_loglik = profile["loglik"].max()
    profile["delta_loglik"] = profile["loglik"] - max_loglik
    profile["relative_likelihood"] = np.exp(profile["delta_loglik"])
    return profile


def download_returns(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance first: python -m pip install yfinance") from exc

    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if prices.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    close = prices["Close"].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = np.log(close / close.shift(1)).dropna()
    return 100.0 * returns.to_numpy(dtype=float)


def fit_normal_student_nuvm(data, seed=2026):
    data = np.asarray(data, dtype=float)

    normal_mu, normal_sigma = stats.norm.fit(data)
    normal_loglik = float(np.sum(stats.norm.logpdf(data, loc=normal_mu, scale=normal_sigma)))

    t_df, t_loc, t_scale = stats.t.fit(data)
    t_loglik = float(np.sum(stats.t.logpdf(data, df=t_df, loc=t_loc, scale=t_scale)))

    mle = nuvm.estimate_nuvm_mle(data, initial_a=0.5, n_starts=12, maxiter=3000, seed=seed)
    mom = method_of_moments_nuvm(data)

    n_obs = len(data)
    rows = [
        {
            "Model": "Normal",
            "mu": normal_mu,
            "sigma": normal_sigma,
            "shape": np.nan,
            "LogLik": normal_loglik,
            "AIC": nuvm.aic(normal_loglik, 2),
            "BIC": nuvm.bic(normal_loglik, 2, n_obs),
        },
        {
            "Model": "Student-t",
            "mu": t_loc,
            "sigma": t_scale,
            "shape": t_df,
            "LogLik": t_loglik,
            "AIC": nuvm.aic(t_loglik, 3),
            "BIC": nuvm.bic(t_loglik, 3, n_obs),
        },
        {
            "Model": "NUVM-MLE",
            "mu": mle["mu_hat"],
            "sigma": mle["sigma_hat"],
            "shape": mle["a_hat"],
            "LogLik": mle["loglik"],
            "AIC": nuvm.aic(mle["loglik"], 3),
            "BIC": nuvm.bic(mle["loglik"], 3, n_obs),
        },
        {
            "Model": "NUVM-MoM",
            "mu": mom["mu_mom"],
            "sigma": mom["sigma_mom"],
            "shape": mom["a_mom"],
            "LogLik": nuvm.nuvm_loglik((mom["mu_mom"], mom["sigma_mom"], mom["a_mom"]), data),
            "AIC": np.nan,
            "BIC": np.nan,
        },
    ]
    return pd.DataFrame(rows), mle, mom


def run_mom_on_existing_nifty():
    fit_path = OUTPUT_DIR / "05_NUVM_real_data_parameter_estimates.csv"
    if not fit_path.exists():
        raise FileNotFoundError("Run the real-data study first, or use --part multi to download data.")

    table = pd.read_csv(fit_path)
    print("\nExisting fitted parameter table:")
    print(table.to_string(index=False))
    print("\nMoM needs raw returns, so use --part multi or --part profile to compute it from downloaded data.")


def run_profile_for_ticker(ticker="^NSEI", label="NIFTY", start="2015-01-01", end="2025-01-01"):
    data = download_returns(ticker, start, end)
    profile = profile_loglik_a(data)
    mom = method_of_moments_nuvm(data)
    mle = nuvm.estimate_nuvm_mle(data, initial_a=0.5, n_starts=12, maxiter=3000)

    profile_path = OUTPUT_DIR / f"NUVM_likelihood_profile_{label}.csv"
    mom_path = OUTPUT_DIR / f"NUVM_method_of_moments_{label}.csv"
    profile.to_csv(profile_path, index=False)
    pd.DataFrame([mom]).to_csv(mom_path, index=False)

    print(f"\n{label} NUVM MLE:")
    print(pd.DataFrame([mle]).drop(columns=["message"], errors="ignore").to_string(index=False))
    print(f"\n{label} Method of Moments:")
    print(pd.DataFrame([mom]).to_string(index=False))
    print(f"\n{label} likelihood profile top rows:")
    print(profile.sort_values("loglik", ascending=False).head(10).to_string(index=False))
    print("\nSaved:", profile_path)
    print("Saved:", mom_path)

    make_likelihood_profile_plot(profile, label)
    make_density_and_qq_plots(data, label, mle)
    return profile, mom, mle


def run_multi_dataset_study(start="2015-01-01", end="2025-01-01"):
    datasets = {
        "NIFTY": "^NSEI",
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "GOLD": "GC=F",
        "CRUDE": "CL=F",
    }
    all_rows = []

    for label, ticker in datasets.items():
        print(f"\nDownloading and fitting {label}: {ticker}")
        data = download_returns(ticker, start, end)
        comparison, mle, mom = fit_normal_student_nuvm(data)
        comparison.insert(0, "Dataset", label)
        comparison.insert(1, "Ticker", ticker)
        comparison.insert(2, "n", len(data))
        comparison.insert(3, "SampleKurtosis", stats.kurtosis(data, fisher=False))
        comparison.insert(4, "SampleSkewness", stats.skew(data))
        all_rows.append(comparison)

    result = pd.concat(all_rows, ignore_index=True)
    path = OUTPUT_DIR / "NUVM_multi_dataset_comparison.csv"
    result.to_csv(path, index=False)
    print("\nMulti-dataset comparison:")
    print(result.to_string(index=False))
    print("\nSaved:", path)
    return result


def run_a_recovery_study(replications=1000, n=500, seed=2026):
    true_mu = 0.0
    true_sigma = 1.0
    true_a_values = [0.2, 0.5, 0.8]
    rng = np.random.default_rng(seed)
    rows = []

    for true_a in true_a_values:
        print(f"\nRecovery study for true a={true_a}")
        for r in range(1, replications + 1):
            sample = nuvm.generate_nuvm(n=n, mu=true_mu, sigma=true_sigma, a=true_a, rng=rng)
            fit = nuvm.estimate_nuvm_mle(sample, initial_a=true_a, n_starts=8, seed=seed + r)
            mom = method_of_moments_nuvm(sample)
            rows.append(
                {
                    "true_a": true_a,
                    "replication": r,
                    "success": fit["success"],
                    "mle_a_hat": fit["a_hat"],
                    "mom_a_hat": mom["a_mom"],
                    "mle_mu_hat": fit["mu_hat"],
                    "mle_sigma_hat": fit["sigma_hat"],
                }
            )
            if r % max(1, replications // 10) == 0:
                print(f"  completed {r}/{replications}")

    raw = pd.DataFrame(rows)
    summary_rows = []
    for true_a, group in raw.groupby("true_a"):
        summary_rows.append(
            {
                "True a": true_a,
                "Mean MLE a_hat": group["mle_a_hat"].mean(),
                "RMSE MLE a_hat": np.sqrt(np.mean((group["mle_a_hat"] - true_a) ** 2)),
                "Mean MoM a_hat": group["mom_a_hat"].mean(),
                "RMSE MoM a_hat": np.sqrt(np.mean((group["mom_a_hat"] - true_a) ** 2)),
                "Convergence Rate": group["success"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    raw_path = OUTPUT_DIR / "NUVM_a_recovery_raw.csv"
    summary_path = OUTPUT_DIR / "NUVM_a_recovery_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("\na recovery summary:")
    print(summary.to_string(index=False))
    print("\nSaved:", raw_path)
    print("Saved:", summary_path)
    return raw, summary


def make_likelihood_profile_plot(profile, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile["a"], profile["loglik"], marker="o", linewidth=1.8)
    ax.set_xlabel("a")
    ax.set_ylabel("Profile log-likelihood")
    ax.set_title(f"NUVM likelihood profile for a: {label}")
    ax.grid(alpha=0.25)
    path = FIGURE_DIR / f"NUVM_likelihood_profile_{label}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("Saved:", path)


def make_density_and_qq_plots(data, label, nuvm_fit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    data = np.asarray(data, dtype=float)
    normal_mu, normal_sigma = stats.norm.fit(data)
    t_df, t_loc, t_scale = stats.t.fit(data)

    x_grid = np.linspace(np.quantile(data, 0.005), np.quantile(data, 0.995), 600)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data, bins=50, density=True, alpha=0.35, color="#8aa6c1", edgecolor="white")
    ax.plot(x_grid, stats.norm.pdf(x_grid, normal_mu, normal_sigma), label="Normal", linewidth=2)
    ax.plot(x_grid, stats.t.pdf(x_grid, t_df, t_loc, t_scale), label="Student-t", linewidth=2)
    ax.plot(
        x_grid,
        nuvm.nuvm_pdf(x_grid, nuvm_fit["mu_hat"], nuvm_fit["sigma_hat"], nuvm_fit["a_hat"]),
        label="NUVM",
        linewidth=2,
    )
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(f"Histogram with fitted densities: {label}")
    ax.legend()
    ax.grid(alpha=0.2)
    hist_path = FIGURE_DIR / f"NUVM_histogram_densities_{label}.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=300)
    plt.close(fig)

    sorted_data = np.sort(data)
    probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    normal_q = stats.norm.ppf(probs, normal_mu, normal_sigma)
    t_q = stats.t.ppf(probs, t_df, t_loc, t_scale)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].scatter(normal_q, sorted_data, s=8, alpha=0.65)
    axes[0].set_title("Normal QQ")
    axes[1].scatter(t_q, sorted_data, s=8, alpha=0.65)
    axes[1].set_title("Student-t QQ")
    for ax in axes:
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.grid(alpha=0.2)
    qq_path = FIGURE_DIR / f"NUVM_QQ_plots_{label}.png"
    fig.tight_layout()
    fig.savefig(qq_path, dpi=300)
    plt.close(fig)

    print("Saved:", hist_path)
    print("Saved:", qq_path)


def make_rmse_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = OUTPUT_DIR / "05_NUVM_simulation_summary.csv"
    if not summary_path.exists():
        print("Skipping RMSE plot because 05_NUVM_simulation_summary.csv was not found.")
        return None

    summary = pd.read_csv(summary_path)
    FIGURE_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column in ["RMSE(mu)", "RMSE(sigma)", "RMSE(a)"]:
        if column in summary.columns:
            ax.plot(summary["n"], summary[column], marker="o", label=column)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("RMSE")
    ax.set_title("NUVM MLE RMSE versus sample size")
    ax.legend()
    ax.grid(alpha=0.25)
    path = FIGURE_DIR / "NUVM_RMSE_vs_n.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("Saved:", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="NUVM reviewer diagnostics.")
    parser.add_argument(
        "--part",
        choices=["profile", "mom", "recovery", "multi", "figures", "all"],
        default="profile",
    )
    parser.add_argument("--ticker", default="^NSEI", help="Ticker for profile/figures.")
    parser.add_argument("--label", default="NIFTY", help="Short label for output files.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.part == "mom":
        data = download_returns(args.ticker, args.start, args.end)
        mom = method_of_moments_nuvm(data)
        path = OUTPUT_DIR / f"NUVM_method_of_moments_{args.label}.csv"
        pd.DataFrame([mom]).to_csv(path, index=False)
        print(pd.DataFrame([mom]).to_string(index=False))
        print("\nSaved:", path)

    profile_result = None
    if args.part in {"profile", "all"}:
        profile_result = run_profile_for_ticker(args.ticker, args.label, args.start, args.end)

    if args.part == "figures":
        run_profile_for_ticker(args.ticker, args.label, args.start, args.end)

    if args.part in {"recovery", "all"}:
        run_a_recovery_study(replications=args.replications, n=args.n, seed=args.seed)

    if args.part in {"multi", "all"}:
        run_multi_dataset_study(args.start, args.end)

    if args.part in {"figures", "all"}:
        make_rmse_plot()


if __name__ == "__main__":
    main()
