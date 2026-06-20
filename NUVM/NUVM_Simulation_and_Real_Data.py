"""
NUVM Simulation Study and Real Data Application

This script contains paper-ready empirical sections:

1. Monte Carlo simulation study for NUVM MLE parameter recovery.
2. Real-data application using NIFTY 50 daily log returns.
3. Higher moments, Fisher information, bootstrap confidence intervals,
   and goodness-of-fit statistics.

Run examples:

    python 05_NUVM_Simulation_and_Real_Data.py --part simulation
    python 05_NUVM_Simulation_and_Real_Data.py --part real
    python 05_NUVM_Simulation_and_Real_Data.py --part both

For a quick simulation test:

    python 05_NUVM_Simulation_and_Real_Data.py --part simulation --replications 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import erf


OUTPUT_DIR = Path(__file__).resolve().parent


def nuvm_pdf(x, mu, sigma, a):
    """Numerically stable NUVM density."""
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    if sigma <= 0 or a <= 0 or a >= 1:
        out = np.full(x.shape, 1e-300)
        return float(out[0]) if scalar_input else out

    z = x - mu
    abs_z = np.abs(z)
    L = (1.0 - a) * sigma**2
    U = (1.0 + a) * sigma**2
    width = U - L

    if width <= 1e-12:
        density = stats.norm.pdf(x, loc=mu, scale=sigma)
        return float(density[0]) if scalar_input else density

    def antiderivative(v):
        safe_v = np.maximum(v, 1e-300)
        expo_arg = np.minimum((z**2) / (2.0 * safe_v), 700.0)
        return (
            2.0 * np.sqrt(safe_v) * np.exp(-expo_arg)
            + np.sqrt(2.0 * np.pi) * abs_z * erf(abs_z / np.sqrt(2.0 * safe_v))
        )

    density_at_mu = 2.0 * (np.sqrt(U) - np.sqrt(L)) / (width * np.sqrt(2.0 * np.pi))

    with np.errstate(all="ignore"):
        density = (antiderivative(U) - antiderivative(L)) / (width * np.sqrt(2.0 * np.pi))

    density = np.where(abs_z < 1e-12, density_at_mu, density)
    density = np.maximum(density, 1e-300)
    return float(density[0]) if scalar_input else density


def nuvm_cdf(x, mu, sigma, a, quadrature_points=80):
    """NUVM CDF by Gauss-Legendre integration over the uniform variance."""
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    if sigma <= 0 or a <= 0 or a >= 1:
        out = np.full(x.shape, np.nan)
        return float(out[0]) if scalar_input else out

    L = (1.0 - a) * sigma**2
    U = (1.0 + a) * sigma**2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    v = L + 0.5 * (nodes + 1.0) * (U - L)
    w = 0.5 * weights
    cdf = np.sum(w[:, None] * stats.norm.cdf((x[None, :] - mu) / np.sqrt(v[:, None])), axis=0)
    cdf = np.clip(cdf, 0.0, 1.0)
    return float(cdf[0]) if scalar_input else cdf


def nuvm_loglik(params, data):
    """NUVM log-likelihood for params=(mu, sigma, a)."""
    mu, sigma, a = params
    if sigma <= 0 or a <= 0 or a >= 1:
        return -np.inf
    pdf_values = nuvm_pdf(data, mu, sigma, a)
    if np.any(~np.isfinite(pdf_values)) or np.any(pdf_values <= 0):
        return -np.inf
    return float(np.sum(np.log(pdf_values)))


def negative_loglik_bounded(params, data):
    """Negative log-likelihood using direct bounded parameters."""
    value = nuvm_loglik(params, data)
    if not np.isfinite(value):
        return 1e100
    return -value


def nuvm_raw_variance_moment(power, sigma, a):
    """E[V^power] for V ~ Uniform((1-a)sigma^2, (1+a)sigma^2)."""
    L = (1.0 - a) * sigma**2
    U = (1.0 + a) * sigma**2
    return (U ** (power + 1) - L ** (power + 1)) / ((power + 1) * (U - L))


def nuvm_even_central_moment(order, sigma, a):
    """E[(X-mu)^order] for even orders under the NUVM model."""
    if order % 2 != 0 or order < 2:
        raise ValueError("order must be an even integer greater than or equal to 2")
    normal_constant = 1
    for value in range(1, order, 2):
        normal_constant *= value
    return normal_constant * nuvm_raw_variance_moment(order // 2, sigma, a)


def nuvm_higher_moments(mu, sigma, a):
    """Paper-ready sixth and eighth central moments."""
    return {
        "mu": mu,
        "sigma": sigma,
        "a": a,
        "E[V^3]": sigma**6 * (1.0 + a**2),
        "Sixth central moment": 15.0 * sigma**6 * (1.0 + a**2),
        "E[V^4]": sigma**8 * (1.0 + 2.0 * a**2 + a**4 / 5.0),
        "Eighth central moment": 105.0 * sigma**8 * (1.0 + 2.0 * a**2 + a**4 / 5.0),
    }


def mom_start_nuvm(data):
    """Stable method-of-moments style starting values for MLE."""
    data = np.asarray(data, dtype=float)
    mu_0 = float(np.mean(data))
    sigma_0 = max(float(np.std(data, ddof=0)), 1e-4)
    centered = data - mu_0
    variance_0 = float(np.mean(centered**2))

    if variance_0 <= 0:
        a_0 = 0.3
    else:
        raw_kurtosis = float(np.mean(centered**4) / variance_0**2)
        a_0 = np.sqrt(max(raw_kurtosis - 3.0, 0.0))

    return np.array([mu_0, sigma_0, float(np.clip(a_0, 0.02, 0.95))])


def estimate_nuvm_mle(data, initial_a=0.5, maxiter=3000, n_starts=8, seed=123):
    """
    Estimate NUVM parameters by MLE.

    This corrected version uses direct parameters and L-BFGS-B bounds instead
    of exp/logistic transforms, avoiding overflow in sigma and a.
    """
    data = np.asarray(data, dtype=float)
    initial = mom_start_nuvm(data)
    initial[2] = float(np.clip(initial_a if initial_a is not None else initial[2], 1e-4, 0.999))

    data_scale = max(float(np.std(data, ddof=0)), 1e-4)
    bounds = [
        (float(np.min(data) - 5.0 * data_scale), float(np.max(data) + 5.0 * data_scale)),
        (1e-6, 10.0 * data_scale),
        (1e-4, 0.999),
    ]

    rng = np.random.default_rng(seed)
    starts = [initial.copy(), mom_start_nuvm(data)]
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(
            np.array(
                [
                    initial[0] + rng.normal(0.0, 0.25 * data_scale),
                    max(initial[1] * rng.uniform(0.5, 1.8), 1e-5),
                    rng.uniform(0.05, 0.95),
                ]
            )
        )

    best = None
    best_fun = np.inf
    for start in starts:
        start = np.array(
            [
                np.clip(start[0], bounds[0][0], bounds[0][1]),
                np.clip(start[1], bounds[1][0], bounds[1][1]),
                np.clip(start[2], bounds[2][0], bounds[2][1]),
            ]
        )
        try:
            result = minimize(
                negative_loglik_bounded,
                start,
                args=(data,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
            )
        except Exception as exc:
            result = None

        if result is not None and np.isfinite(result.fun) and result.fun < best_fun:
            best = result
            best_fun = float(result.fun)

    if best is None:
        return {
            "mu_hat": np.nan,
            "sigma_hat": np.nan,
            "a_hat": np.nan,
            "loglik": -np.inf,
            "success": False,
            "message": "All bounded L-BFGS-B starts failed",
        }

    valid = np.isfinite(best.fun) and best.x[1] > 0 and 0 < best.x[2] < 1
    return {
        "mu_hat": float(best.x[0]),
        "sigma_hat": float(best.x[1]),
        "a_hat": float(best.x[2]),
        "loglik": float(-best.fun),
        "success": bool(valid),
        "message": str(best.message),
    }


def generate_nuvm(n, mu, sigma, a, rng):
    """Generate an iid NUVM sample."""
    L = (1.0 - a) * sigma**2
    U = (1.0 + a) * sigma**2
    v = rng.uniform(L, U, size=n)
    return rng.normal(mu, np.sqrt(v))


def bias(values, truth):
    return np.mean(values) - truth


def rmse(values, truth):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean((values - truth) ** 2))


def run_simulation_study(replications=1000, seed=2026):
    """Run the NUVM MLE simulation study."""
    true_mu = 0.0
    true_sigma = 1.0
    true_a = 0.5
    sample_sizes = [50, 100, 250, 500, 1000]
    rng = np.random.default_rng(seed)

    rows = []
    total_fits = len(sample_sizes) * replications
    print("Simulation study")
    print("True parameters:", true_mu, true_sigma, true_a)
    print("Sample sizes:", sample_sizes)
    print("Replications per sample size:", replications)
    print("Total MLE fits:", total_fits)

    for n in sample_sizes:
        print(f"\nStarting n={n}")
        for r in range(1, replications + 1):
            sample = generate_nuvm(n=n, mu=true_mu, sigma=true_sigma, a=true_a, rng=rng)
            estimate = estimate_nuvm_mle(sample, seed=seed + n * 100000 + r)
            estimate.update({"n": n, "replication": r})
            rows.append(estimate)

            if r % max(1, replications // 10) == 0:
                print(f"  completed {r}/{replications}")

    results = pd.DataFrame(rows)
    successful = results["success"].astype(bool)
    summary_rows = []

    for n, group in results.groupby("n"):
        group_success = group[group["success"].astype(bool)]
        summary_group = group_success if not group_success.empty else group
        summary_rows.append(
            {
                "n": n,
                "Bias(mu)": bias(summary_group["mu_hat"], true_mu),
                "RMSE(mu)": rmse(summary_group["mu_hat"], true_mu),
                "Bias(sigma)": bias(summary_group["sigma_hat"], true_sigma),
                "RMSE(sigma)": rmse(summary_group["sigma_hat"], true_sigma),
                "Bias(a)": bias(summary_group["a_hat"], true_a),
                "RMSE(a)": rmse(summary_group["a_hat"], true_a),
                "Convergence Rate": np.mean(group["success"]),
                "Successful Fits": int(group["success"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)

    raw_path = OUTPUT_DIR / "05_NUVM_simulation_raw_results.csv"
    summary_path = OUTPUT_DIR / "05_NUVM_simulation_summary.csv"
    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nOverall successful fits:", int(successful.sum()), "/", len(successful))
    print("\nSimulation summary:")
    print(summary.to_string(index=False))
    print("\nSaved:", raw_path)
    print("Saved:", summary_path)

    return results, summary


def aic(loglik, k):
    return 2 * k - 2 * loglik


def bic(loglik, k, n):
    return k * np.log(n) - 2 * loglik


def nuvm_tail_probability(c, sigma, a):
    """P(|X-mu| > c*sigma) for NUVM."""
    threshold = c * sigma
    L = (1.0 - a) * sigma**2
    U = (1.0 + a) * sigma**2
    g = 1.0 / (U - L)

    def integrand(v):
        return 2.0 * stats.norm.sf(threshold / np.sqrt(v)) * g

    value, _ = quad(integrand, L, U, epsabs=1e-12, epsrel=1e-12)
    return value


def finite_difference_hessian(func, params, step=None):
    """Central finite-difference Hessian for a scalar objective."""
    params = np.asarray(params, dtype=float)
    n_params = len(params)
    if step is None:
        step = 1e-4 * np.maximum(np.abs(params), 1.0)
    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        for j in range(i, n_params):
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = step[i]
            ej[j] = step[j]
            fpp = func(params + ei + ej)
            fpm = func(params + ei - ej)
            fmp = func(params - ei + ej)
            fmm = func(params - ei - ej)
            value = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def fisher_information_and_ci(data, params, alpha=0.05):
    """Observed Fisher information, standard errors, and Wald CIs."""
    params = np.asarray(params, dtype=float)

    def objective(theta):
        return negative_loglik_bounded(theta, data)

    info = finite_difference_hessian(objective, params)
    try:
        covariance = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(info)

    covariance = np.asarray(covariance, dtype=float)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = np.column_stack([params - z * se, params + z * se])
    ci[1, 0] = max(ci[1, 0], 0.0)
    ci[2, :] = np.clip(ci[2, :], 0.0, 1.0)

    return info, pd.DataFrame(
        {
            "parameter": ["mu", "sigma", "a"],
            "estimate": params,
            "std_error": se,
            "ci_lower": ci[:, 0],
            "ci_upper": ci[:, 1],
        }
    )


def bootstrap_nuvm_ci(data, B=1000, seed=2026, confidence=0.95):
    """Nonparametric bootstrap CIs for NUVM MLE parameters."""
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(1, B + 1):
        sample = rng.choice(data, size=len(data), replace=True)
        fit = estimate_nuvm_mle(sample, seed=seed + b, n_starts=5, maxiter=1500)
        rows.append(
            {
                "bootstrap": b,
                "success": fit["success"],
                "mu": fit["mu_hat"],
                "sigma": fit["sigma_hat"],
                "a": fit["a_hat"],
            }
        )
        if b % max(1, B // 10) == 0:
            print(f"  bootstrap completed {b}/{B}")

    estimates = pd.DataFrame(rows)
    successful = estimates[estimates["success"].astype(bool)]
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q

    ci_rows = []
    for param in ["mu", "sigma", "a"]:
        values = successful[param].dropna()
        ci_rows.append(
            {
                "parameter": param,
                "bootstrap_successes": len(successful),
                "B": B,
                "mean": values.mean(),
                "std_error": values.std(ddof=1),
                "ci_lower": values.quantile(lower_q),
                "ci_upper": values.quantile(upper_q),
            }
        )

    return estimates, pd.DataFrame(ci_rows)


def gof_statistics_from_cdf(data, cdf_values):
    """KS, Anderson-Darling, and Cramer-von Mises statistics."""
    values = np.sort(np.asarray(cdf_values, dtype=float))
    values = np.clip(values, 1e-12, 1.0 - 1e-12)
    n = len(values)
    i = np.arange(1, n + 1)
    ks = np.max(np.maximum(i / n - values, values - (i - 1) / n))
    ad = -n - np.mean((2 * i - 1) * (np.log(values) + np.log(1.0 - values[::-1])))
    cvm = 1.0 / (12.0 * n) + np.sum((values - (2 * i - 1) / (2.0 * n)) ** 2)
    return ks, ad, cvm


def goodness_of_fit_table(data, normal_params, t_params, nuvm_params):
    """Goodness-of-fit table for Normal, Student-t, and NUVM."""
    normal_mu, normal_sigma = normal_params
    t_df, t_loc, t_scale = t_params
    nuvm_mu, nuvm_sigma, nuvm_a = nuvm_params
    sorted_data = np.sort(data)

    model_cdfs = {
        "Normal": stats.norm.cdf(sorted_data, loc=normal_mu, scale=normal_sigma),
        "Student-t": stats.t.cdf(sorted_data, df=t_df, loc=t_loc, scale=t_scale),
        "NUVM": nuvm_cdf(sorted_data, nuvm_mu, nuvm_sigma, nuvm_a),
    }

    rows = []
    for model, cdf_values in model_cdfs.items():
        ks, ad, cvm = gof_statistics_from_cdf(sorted_data, cdf_values)
        rows.append({"Model": model, "KS": ks, "AD": ad, "CvM": cvm})

    return pd.DataFrame(rows)


def run_real_data_study(start="2015-01-01", end="2025-01-01", bootstrap_replications=1000):
    """Fit Normal, Student-t, and NUVM models to NIFTY 50 returns."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the real-data study. Install it with: "
            "python -m pip install yfinance"
        ) from exc

    ticker = "^NSEI"
    print("Downloading:", ticker, start, end)
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if data.empty:
        raise RuntimeError("No data downloaded. Check internet access or ticker availability.")

    close = data["Close"].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = np.log(close / close.shift(1)).dropna()
    r = 100.0 * returns.to_numpy(dtype=float)

    print("Number of returns:", len(r))
    print("Mean:", np.mean(r))
    print("Std:", np.std(r, ddof=0))
    print("Skewness:", stats.skew(r))
    print("Kurtosis:", stats.kurtosis(r, fisher=False))

    normal_mu, normal_sigma = stats.norm.fit(r)
    normal_loglik = float(np.sum(stats.norm.logpdf(r, loc=normal_mu, scale=normal_sigma)))

    t_df, t_loc, t_scale = stats.t.fit(r)
    t_loglik = float(np.sum(stats.t.logpdf(r, df=t_df, loc=t_loc, scale=t_scale)))

    nuvm_fit = estimate_nuvm_mle(r, initial_a=0.5, maxiter=3000, n_starts=12)
    nuvm_mu = nuvm_fit["mu_hat"]
    nuvm_sigma = nuvm_fit["sigma_hat"]
    nuvm_a = nuvm_fit["a_hat"]
    nuvm_loglik_value = nuvm_fit["loglik"]

    n_obs = len(r)
    comparison = pd.DataFrame(
        [
            {
                "Model": "Normal",
                "k": 2,
                "LogLik": normal_loglik,
                "AIC": aic(normal_loglik, 2),
                "BIC": bic(normal_loglik, 2, n_obs),
            },
            {
                "Model": "Student-t",
                "k": 3,
                "LogLik": t_loglik,
                "AIC": aic(t_loglik, 3),
                "BIC": bic(t_loglik, 3, n_obs),
            },
            {
                "Model": "NUVM",
                "k": 3,
                "LogLik": nuvm_loglik_value,
                "AIC": aic(nuvm_loglik_value, 3),
                "BIC": bic(nuvm_loglik_value, 3, n_obs),
            },
        ]
    )

    empirical_mu = np.mean(r)
    empirical_sigma = np.std(r, ddof=0)
    tail_rows = []

    for c in [2, 3, 4]:
        empirical = np.mean(np.abs(r - empirical_mu) > c * empirical_sigma)
        normal_tail = 2.0 * stats.norm.sf(c)
        t_tail = (
            stats.t.sf(empirical_mu + c * empirical_sigma, df=t_df, loc=t_loc, scale=t_scale)
            + stats.t.cdf(empirical_mu - c * empirical_sigma, df=t_df, loc=t_loc, scale=t_scale)
        )
        nuvm_tail = nuvm_tail_probability(c, nuvm_sigma, nuvm_a)
        tail_rows.append(
            {
                "c": c,
                "Empirical": empirical,
                "Normal": normal_tail,
                "Student-t": t_tail,
                "NUVM": nuvm_tail,
            }
        )

    tail_table = pd.DataFrame(tail_rows)
    fit_table = pd.DataFrame(
        [
            {"Model": "Normal", "mu": normal_mu, "sigma": normal_sigma, "a/df": np.nan},
            {"Model": "Student-t", "mu": t_loc, "sigma": t_scale, "a/df": t_df},
            {"Model": "NUVM", "mu": nuvm_mu, "sigma": nuvm_sigma, "a/df": nuvm_a},
        ]
    )

    moment_table = pd.DataFrame([nuvm_higher_moments(nuvm_mu, nuvm_sigma, nuvm_a)])
    fisher_matrix, fisher_ci = fisher_information_and_ci(r, [nuvm_mu, nuvm_sigma, nuvm_a])
    fisher_table = pd.DataFrame(
        fisher_matrix,
        index=["mu", "sigma", "a"],
        columns=["mu", "sigma", "a"],
    )
    gof_table = goodness_of_fit_table(
        r,
        normal_params=(normal_mu, normal_sigma),
        t_params=(t_df, t_loc, t_scale),
        nuvm_params=(nuvm_mu, nuvm_sigma, nuvm_a),
    )

    bootstrap_estimates = pd.DataFrame()
    bootstrap_ci = pd.DataFrame()
    if bootstrap_replications > 0:
        print(f"\nBootstrap NUVM confidence intervals, B={bootstrap_replications}")
        bootstrap_estimates, bootstrap_ci = bootstrap_nuvm_ci(r, B=bootstrap_replications)

    comparison_path = OUTPUT_DIR / "05_NUVM_real_data_model_comparison.csv"
    tail_path = OUTPUT_DIR / "05_NUVM_real_data_tail_comparison.csv"
    fit_path = OUTPUT_DIR / "05_NUVM_real_data_parameter_estimates.csv"
    moment_path = OUTPUT_DIR / "05_NUVM_higher_moments.csv"
    fisher_path = OUTPUT_DIR / "05_NUVM_observed_fisher_information.csv"
    fisher_ci_path = OUTPUT_DIR / "05_NUVM_wald_confidence_intervals.csv"
    gof_path = OUTPUT_DIR / "05_NUVM_goodness_of_fit.csv"
    bootstrap_raw_path = OUTPUT_DIR / "05_NUVM_bootstrap_raw_results.csv"
    bootstrap_ci_path = OUTPUT_DIR / "05_NUVM_bootstrap_confidence_intervals.csv"

    comparison.to_csv(comparison_path, index=False)
    tail_table.to_csv(tail_path, index=False)
    fit_table.to_csv(fit_path, index=False)
    moment_table.to_csv(moment_path, index=False)
    fisher_table.to_csv(fisher_path)
    fisher_ci.to_csv(fisher_ci_path, index=False)
    gof_table.to_csv(gof_path, index=False)
    if bootstrap_replications > 0:
        bootstrap_estimates.to_csv(bootstrap_raw_path, index=False)
        bootstrap_ci.to_csv(bootstrap_ci_path, index=False)

    print("\nParameter estimates:")
    print(fit_table.to_string(index=False))
    print("\nModel comparison:")
    print(comparison.sort_values("AIC").to_string(index=False))
    print("\nTail comparison:")
    print(tail_table.to_string(index=False))
    print("\nHigher moments:")
    print(moment_table.to_string(index=False))
    print("\nObserved Fisher/Wald confidence intervals:")
    print(fisher_ci.to_string(index=False))
    print("\nGoodness-of-fit:")
    print(gof_table.to_string(index=False))
    if bootstrap_replications > 0:
        print("\nBootstrap confidence intervals:")
        print(bootstrap_ci.to_string(index=False))

    print("\nSaved:", comparison_path)
    print("Saved:", tail_path)
    print("Saved:", fit_path)
    print("Saved:", moment_path)
    print("Saved:", fisher_path)
    print("Saved:", fisher_ci_path)
    print("Saved:", gof_path)
    if bootstrap_replications > 0:
        print("Saved:", bootstrap_raw_path)
        print("Saved:", bootstrap_ci_path)

    return comparison, tail_table, fit_table, moment_table, fisher_ci, gof_table, bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="NUVM simulation and real-data studies.")
    parser.add_argument(
        "--part",
        choices=["simulation", "real", "both"],
        default="both",
        help="Which section to run.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=1000,
        help="Monte Carlo replications per sample size.",
    )
    parser.add_argument(
        "--bootstrap-replications",
        type=int,
        default=1000,
        help="Bootstrap replications for real-data NUVM confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--start", default="2015-01-01", help="Real-data start date.")
    parser.add_argument("--end", default="2025-01-01", help="Real-data end date.")
    args = parser.parse_args()

    if args.part in {"simulation", "both"}:
        run_simulation_study(replications=args.replications, seed=args.seed)

    if args.part in {"real", "both"}:
        run_real_data_study(
            start=args.start,
            end=args.end,
            bootstrap_replications=args.bootstrap_replications,
        )


if __name__ == "__main__":
    main()
