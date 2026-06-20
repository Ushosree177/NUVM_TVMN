"""
TVMN Empirical Studies

This script covers the next four paper sections:

    1. Fisher information, standard errors, and 95% Wald confidence intervals.
    2. Monte Carlo parameter recovery study.
    3. Real-data applications comparing Normal, Laplace, NUVM, and TVMN.
    4. Parametric bootstrap confidence intervals.

Run examples from:

    C:\\Users\\LENOVO\\Documents\\distribution

Quick verification run:

    python TVMN\\04_TVMN_empirical_studies.py --part all --mc-replications 5 --bootstrap-replications 20

Paper-scale Monte Carlo:

    python TVMN\\04_TVMN_empirical_studies.py --part monte-carlo --mc-replications 1000

Paper-scale bootstrap:

    python TVMN\\04_TVMN_empirical_studies.py --part bootstrap --bootstrap-replications 1000

Real-data applications:

    python TVMN\\04_TVMN_empirical_studies.py --part real-data
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn import datasets


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_module(module_name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tvmn_core = load_module("tvmn_core_empirical", BASE_DIR / "01_TVMN_step_by_step_derivation.py")
tvmn_mle = load_module("tvmn_mle_empirical", BASE_DIR / "03_TVMN_likelihood_mle.py")
nuvm_core = load_module("nuvm_core_empirical", PROJECT_DIR / "NUVM" / "NUVM_MLE.py")


def ensure_output_dirs():
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def estimate_tvmn(data, n_starts=18, seed=123):
    """Wrapper returning native TVMN estimates as a vector and dict."""
    fit = tvmn_mle.estimate_tvmn_mle(data, n_starts=n_starts, seed=seed)
    params = np.array([fit["a_hat"], fit["m_hat"], fit["b_hat"]], dtype=float)
    return params, fit


def numerical_hessian(func, theta, step=1e-4):
    """Central-difference Hessian for a scalar function."""
    theta = np.asarray(theta, dtype=float)
    p = len(theta)
    hessian = np.zeros((p, p), dtype=float)

    for i in range(p):
        for j in range(p):
            ei = np.zeros(p)
            ej = np.zeros(p)
            ei[i] = step
            ej[j] = step
            f_pp = func(theta + ei + ej)
            f_pm = func(theta + ei - ej)
            f_mp = func(theta - ei + ej)
            f_mm = func(theta - ei - ej)
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)

    return 0.5 * (hessian + hessian.T)


def theta_jacobian_native(theta):
    """
    Jacobian of native params (a,m,b) with respect to theta.

    a = exp(theta0)
    m = a + exp(theta1)
    b = m + exp(theta2)
    """
    gap0, gap1, gap2 = np.exp(theta)
    return np.array(
        [
            [gap0, 0.0, 0.0],
            [gap0, gap1, 0.0],
            [gap0, gap1, gap2],
        ],
        dtype=float,
    )


def observed_fisher_information(data, params_hat):
    """
    Observed Fisher information using the Hessian of the negative log-likelihood.

    The Hessian is computed in unconstrained theta space. Standard errors are
    moved to native (a,m,b) space by the delta method.
    """
    theta_hat = tvmn_mle.params_to_theta(params_hat)
    sample_variance = max(float(np.var(data, ddof=0)), 1e-4)
    max_square = max(float(np.max(np.asarray(data) ** 2)), sample_variance)
    upper_b = max(20.0 * sample_variance, 4.0 * max_square, 5.0)

    objective = lambda theta: tvmn_mle.neg_loglik_theta(theta, data, upper_b)
    hessian_theta = numerical_hessian(objective, theta_hat, step=1e-4)

    ridge = 1e-6 * np.eye(3)
    try:
        cov_theta = np.linalg.inv(hessian_theta + ridge)
    except np.linalg.LinAlgError:
        cov_theta = np.linalg.pinv(hessian_theta + ridge)

    jacobian = theta_jacobian_native(theta_hat)
    cov_native = jacobian @ cov_theta @ jacobian.T
    se_native = np.sqrt(np.maximum(np.diag(cov_native), 0.0))

    ci_lower = params_hat - 1.96 * se_native
    ci_upper = params_hat + 1.96 * se_native

    return {
        "hessian_theta": hessian_theta,
        "cov_theta": cov_theta,
        "cov_native": cov_native,
        "se_native": se_native,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def run_fisher_section(n=500, true_params=(0.5, 1.0, 2.0), seed=777):
    """Fit one artificial dataset and compute observed Fisher information."""
    data = tvmn_mle.generate_tvmn_sample(n, *true_params, seed=seed)
    params_hat, fit = estimate_tvmn(data, n_starts=24, seed=seed + 1)
    fisher = observed_fisher_information(data, params_hat)

    names = ["a", "m", "b"]
    rows = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "parameter": name,
                "estimate": params_hat[idx],
                "std_error": fisher["se_native"][idx],
                "ci_lower": fisher["ci_lower"][idx],
                "ci_upper": fisher["ci_upper"][idx],
                "true_value": true_params[idx],
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "04_TVMN_fisher_standard_errors_ci.csv", index=False)
    pd.DataFrame(fisher["hessian_theta"], columns=names, index=names).to_csv(
        OUTPUT_DIR / "04_TVMN_observed_fisher_hessian_theta.csv"
    )
    pd.DataFrame(fisher["cov_native"], columns=names, index=names).to_csv(
        OUTPUT_DIR / "04_TVMN_covariance_native_delta_method.csv"
    )

    return fit, fisher


def run_monte_carlo(replications=100, sample_sizes=(30, 50, 100, 200, 500), true_params=(0.5, 1.0, 2.0), seed=2026):
    """Monte Carlo study: bias, MSE, and RMSE for TVMN MLE."""
    rng = np.random.default_rng(seed)
    raw_rows = []

    for n in sample_sizes:
        print(f"Monte Carlo n={n}, replications={replications}")
        for rep in range(1, replications + 1):
            sample_seed = int(rng.integers(1, 2_000_000_000))
            data = tvmn_mle.generate_tvmn_sample(n, *true_params, seed=sample_seed)
            fit = tvmn_mle.estimate_tvmn_mle(data, n_starts=10, seed=sample_seed + 1)
            raw_rows.append(
                {
                    "n": n,
                    "replication": rep,
                    "success": fit["success"],
                    "a_hat": fit["a_hat"],
                    "m_hat": fit["m_hat"],
                    "b_hat": fit["b_hat"],
                    "loglik": fit["loglik"],
                }
            )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(OUTPUT_DIR / "04_TVMN_monte_carlo_raw.csv", index=False)

    summary_rows = []
    for n, group in raw.groupby("n"):
        successful = group[group["success"]].copy()
        row = {"n": n, "replications": len(group), "successful": len(successful)}
        for name, true_value in zip(["a", "m", "b"], true_params):
            estimates = successful[f"{name}_hat"].to_numpy(dtype=float)
            errors = estimates - true_value
            row[f"{name}_mean"] = float(np.mean(estimates)) if len(estimates) else np.nan
            row[f"{name}_bias"] = float(np.mean(errors)) if len(errors) else np.nan
            row[f"{name}_mse"] = float(np.mean(errors**2)) if len(errors) else np.nan
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "04_TVMN_monte_carlo_summary.csv", index=False)
    plot_monte_carlo_rmse(summary)
    return raw, summary


def plot_monte_carlo_rmse(summary):
    """Plot RMSE against sample size for a, m, b."""
    plt.figure(figsize=(8.5, 5.2))
    for name in ["a", "m", "b"]:
        plt.plot(summary["n"], summary[f"{name}_rmse"], marker="o", linewidth=2.0, label=f"{name} RMSE")
    plt.title("TVMN Monte Carlo RMSE")
    plt.xlabel("Sample size")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_TVMN_monte_carlo_rmse.png", dpi=300)
    plt.close()


def nuvm_cdf(x, mu, sigma, alpha, quadrature_points=80):
    """NUVM CDF by Gauss-Legendre integration over the uniform variance."""
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    lower_v = (1.0 - alpha) * sigma * sigma
    upper_v = (1.0 + alpha) * sigma * sigma
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    v = lower_v + 0.5 * (nodes + 1.0) * (upper_v - lower_v)
    w = 0.5 * weights
    cdf = np.sum(w[:, None] * stats.norm.cdf((x[None, :] - mu) / np.sqrt(v[:, None])), axis=0)
    return float(cdf[0]) if scalar_input else np.clip(cdf, 0.0, 1.0)


def empirical_ks(data, cdf_func):
    """One-sample KS statistic using fitted CDF values."""
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    cdf_values = np.clip(cdf_func(x), 0.0, 1.0)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf_values)
    d_minus = np.max(cdf_values - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def load_real_datasets():
    """
    Load local real datasets from sklearn.

    The TVMN model here is zero-centered, so each selected numeric variable is
    centered before fitting.
    """
    loaded = []

    diabetes = datasets.load_diabetes()
    loaded.append(("diabetes_target_centered", diabetes.target - np.mean(diabetes.target)))

    breast = datasets.load_breast_cancer()
    loaded.append(("breast_cancer_mean_radius_centered", breast.data[:, 0] - np.mean(breast.data[:, 0])))
    loaded.append(("breast_cancer_mean_area_centered", breast.data[:, 3] - np.mean(breast.data[:, 3])))

    wine = datasets.load_wine()
    loaded.append(("wine_alcohol_centered", wine.data[:, 0] - np.mean(wine.data[:, 0])))

    linnerud = datasets.load_linnerud()
    loaded.append(("linnerud_weight_centered", linnerud.target[:, 1] - np.mean(linnerud.target[:, 1])))

    return [(name, np.asarray(values, dtype=float)) for name, values in loaded if len(values) >= 20]


def fit_normal(data):
    mu, sigma = stats.norm.fit(data)
    sigma = max(float(sigma), 1e-8)
    loglik = float(np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma)))
    return {"model": "Normal", "params": {"mu": mu, "sigma": sigma}, "loglik": loglik, "k": 2}


def fit_laplace(data):
    loc, scale = stats.laplace.fit(data)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.laplace.logpdf(data, loc=loc, scale=scale)))
    return {"model": "Laplace", "params": {"loc": loc, "scale": scale}, "loglik": loglik, "k": 2}


def fit_nuvm(data, seed=123):
    fit = nuvm_core.fit_nuvm_mle(data, n_starts=8, seed=seed)
    return {
        "model": "NUVM",
        "params": {"mu": fit["mu"], "sigma": fit["sigma"], "alpha": fit["a"]},
        "loglik": fit["loglik"],
        "k": 3,
    }


def fit_tvmn_real(data, seed=123):
    params, fit = estimate_tvmn(data, n_starts=18, seed=seed)
    return {
        "model": "TVMN",
        "params": {"a": params[0], "m": params[1], "b": params[2]},
        "loglik": fit["loglik"],
        "k": 3,
    }


def tvmn_location_loglik(params, data):
    """TVMN log-likelihood with an added location parameter mu."""
    mu, a, m, b = np.asarray(params, dtype=float)
    if not np.isfinite(mu + a + m + b):
        return -np.inf
    return tvmn_mle.loglik([a, m, b], np.asarray(data, dtype=float) - mu)


def theta_to_tvmn_location_params(theta):
    """Convert unconstrained theta to (mu, a, m, b)."""
    mu = float(theta[0])
    gaps = np.exp(np.asarray(theta[1:], dtype=float))
    a = gaps[0]
    m = a + gaps[1]
    b = m + gaps[2]
    return np.array([mu, a, m, b], dtype=float)


def neg_tvmn_location_theta(theta, data, upper_b):
    """Negative log-likelihood for location TVMN in transformed parameters."""
    params = theta_to_tvmn_location_params(theta)
    if params[3] > upper_b:
        return 1e100 + (params[3] - upper_b) ** 2
    value = tvmn_location_loglik(params, data)
    if not np.isfinite(value):
        return 1e100
    return -value


def fit_tvmn_location(data, n_starts=18, seed=123):
    """
    Fit TVMN with a location parameter.

    The original theoretical TVMN is centered. For fair benchmark tables on raw
    positive data, this fit uses Y = mu + X, where X follows centered TVMN.
    """
    data = np.asarray(data, dtype=float)
    mu0 = float(np.mean(data))
    centered = data - mu0
    sample_variance = max(float(np.var(centered, ddof=0)), 1e-4)
    max_square = max(float(np.max(centered * centered)), sample_variance)
    upper_b = max(20.0 * sample_variance, 4.0 * max_square, 5.0)
    min_gap = max(1e-5, 1e-6 * sample_variance)
    data_scale = max(float(np.std(data, ddof=1)), 1e-3)

    theta_bounds = [
        (float(np.min(data) - 3.0 * data_scale), float(np.max(data) + 3.0 * data_scale)),
        (np.log(min_gap), np.log(upper_b)),
        (np.log(min_gap), np.log(upper_b)),
        (np.log(min_gap), np.log(upper_b)),
    ]

    starts = []
    for params in tvmn_mle.moment_based_starts(centered):
        starts.append(np.r_[mu0, tvmn_mle.params_to_theta(params)])

    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        mu_start = mu0 + rng.normal(0.0, 0.2 * data_scale)
        raw = np.sort(rng.uniform(0.05 * sample_variance, 3.0 * sample_variance, size=3))
        starts.append(np.r_[mu_start, tvmn_mle.params_to_theta(raw)])

    best_result = None
    best_nll = np.inf
    for theta0 in starts[:n_starts]:
        result = minimize(
            neg_tvmn_location_theta,
            x0=theta0,
            args=(data, upper_b),
            method="L-BFGS-B",
            bounds=theta_bounds,
            options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-6, "maxls": 50},
        )
        if np.isfinite(result.fun) and result.fun < best_nll:
            best_result = result
            best_nll = float(result.fun)

    if best_result is None:
        return {"model": "TVMN", "params": {"mu": np.nan, "a": np.nan, "m": np.nan, "b": np.nan}, "loglik": -np.inf, "k": 4}

    mu, a, m, b = theta_to_tvmn_location_params(best_result.x)
    return {"model": "TVMN", "params": {"mu": mu, "a": a, "m": m, "b": b}, "loglik": -best_nll, "k": 4}


def model_pdf_cdf(model_fit):
    """Return PDF and CDF callables for a fitted model."""
    model = model_fit["model"]
    params = model_fit["params"]

    if model == "Normal":
        return (
            lambda x: stats.norm.pdf(x, loc=params["mu"], scale=params["sigma"]),
            lambda x: stats.norm.cdf(x, loc=params["mu"], scale=params["sigma"]),
        )
    if model == "Laplace":
        return (
            lambda x: stats.laplace.pdf(x, loc=params["loc"], scale=params["scale"]),
            lambda x: stats.laplace.cdf(x, loc=params["loc"], scale=params["scale"]),
        )
    if model == "NUVM":
        return (
            lambda x: nuvm_core.nuvm_pdf(x, params["mu"], params["sigma"], params["alpha"]),
            lambda x: nuvm_cdf(x, params["mu"], params["sigma"], params["alpha"]),
        )
    if model == "TVMN":
        return (
            lambda x: tvmn_core.tvmn_pdf(x, params["a"], params["m"], params["b"]),
            lambda x: tvmn_core.tvmn_cdf(x, params["a"], params["m"], params["b"]),
        )

    raise ValueError(f"Unknown model: {model}")


def model_pdf_cdf_sf(model_fit):
    """Return PDF, CDF, and survival callables for benchmark models."""
    model = model_fit["model"]
    params = model_fit["params"]

    if model == "Normal":
        return (
            lambda x: stats.norm.pdf(x, loc=params["mu"], scale=params["sigma"]),
            lambda x: stats.norm.cdf(x, loc=params["mu"], scale=params["sigma"]),
            lambda x: stats.norm.sf(x, loc=params["mu"], scale=params["sigma"]),
        )
    if model == "Laplace":
        return (
            lambda x: stats.laplace.pdf(x, loc=params["loc"], scale=params["scale"]),
            lambda x: stats.laplace.cdf(x, loc=params["loc"], scale=params["scale"]),
            lambda x: stats.laplace.sf(x, loc=params["loc"], scale=params["scale"]),
        )
    if model == "StudentT":
        return (
            lambda x: stats.t.pdf(x, df=params["df"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.t.cdf(x, df=params["df"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.t.sf(x, df=params["df"], loc=params["loc"], scale=params["scale"]),
        )
    if model == "Gamma":
        return (
            lambda x: stats.gamma.pdf(x, a=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.gamma.cdf(x, a=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.gamma.sf(x, a=params["shape"], loc=params["loc"], scale=params["scale"]),
        )
    if model == "Lognormal":
        return (
            lambda x: stats.lognorm.pdf(x, s=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.lognorm.cdf(x, s=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.lognorm.sf(x, s=params["shape"], loc=params["loc"], scale=params["scale"]),
        )
    if model == "Weibull":
        return (
            lambda x: stats.weibull_min.pdf(x, c=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.weibull_min.cdf(x, c=params["shape"], loc=params["loc"], scale=params["scale"]),
            lambda x: stats.weibull_min.sf(x, c=params["shape"], loc=params["loc"], scale=params["scale"]),
        )
    if model == "Exponential":
        return (
            lambda x: stats.expon.pdf(x, loc=params["loc"], scale=params["scale"]),
            lambda x: stats.expon.cdf(x, loc=params["loc"], scale=params["scale"]),
            lambda x: stats.expon.sf(x, loc=params["loc"], scale=params["scale"]),
        )
    if model == "NUVM":
        return (
            lambda x: nuvm_core.nuvm_pdf(x, params["mu"], params["sigma"], params["alpha"]),
            lambda x: nuvm_cdf(x, params["mu"], params["sigma"], params["alpha"]),
            lambda x: 1.0 - nuvm_cdf(x, params["mu"], params["sigma"], params["alpha"]),
        )
    if model == "TVMN":
        return (
            lambda x: tvmn_core.tvmn_pdf(np.asarray(x) - params["mu"], params["a"], params["m"], params["b"]),
            lambda x: tvmn_core.tvmn_cdf(np.asarray(x) - params["mu"], params["a"], params["m"], params["b"]),
            lambda x: tvmn_core.tvmn_survival(np.asarray(x) - params["mu"], params["a"], params["m"], params["b"]),
        )

    raise ValueError(f"Unknown model: {model}")


def load_benchmark_datasets():
    """Load raw positive datasets for reviewer-style benchmark comparisons."""
    loaded = []

    diabetes = datasets.load_diabetes()
    loaded.append(("diabetes_target", diabetes.target))

    breast = datasets.load_breast_cancer()
    loaded.append(("breast_cancer_mean_radius", breast.data[:, 0]))
    loaded.append(("breast_cancer_mean_texture", breast.data[:, 1]))
    loaded.append(("breast_cancer_mean_area", breast.data[:, 3]))

    wine = datasets.load_wine()
    loaded.append(("wine_alcohol", wine.data[:, 0]))
    loaded.append(("wine_malic_acid", wine.data[:, 1]))
    loaded.append(("wine_proline", wine.data[:, 12]))

    linnerud = datasets.load_linnerud()
    loaded.append(("linnerud_weight", linnerud.target[:, 1]))

    clean = []
    for name, values in loaded:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values > 0.0]
        if len(values) >= 20 and np.std(values) > 0:
            clean.append((name, values))
    return clean


def fit_student_t(data):
    df, loc, scale = stats.t.fit(data)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale)))
    return {"model": "StudentT", "params": {"df": df, "loc": loc, "scale": scale}, "loglik": loglik, "k": 3}


def fit_gamma(data):
    shape, loc, scale = stats.gamma.fit(data, floc=0.0)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.gamma.logpdf(data, a=shape, loc=loc, scale=scale)))
    return {"model": "Gamma", "params": {"shape": shape, "loc": loc, "scale": scale}, "loglik": loglik, "k": 2}


def fit_lognormal(data):
    shape, loc, scale = stats.lognorm.fit(data, floc=0.0)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.lognorm.logpdf(data, s=shape, loc=loc, scale=scale)))
    return {"model": "Lognormal", "params": {"shape": shape, "loc": loc, "scale": scale}, "loglik": loglik, "k": 2}


def fit_weibull(data):
    shape, loc, scale = stats.weibull_min.fit(data, floc=0.0)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.weibull_min.logpdf(data, c=shape, loc=loc, scale=scale)))
    return {"model": "Weibull", "params": {"shape": shape, "loc": loc, "scale": scale}, "loglik": loglik, "k": 2}


def fit_exponential(data):
    loc, scale = stats.expon.fit(data, floc=0.0)
    scale = max(float(scale), 1e-8)
    loglik = float(np.sum(stats.expon.logpdf(data, loc=loc, scale=scale)))
    return {"model": "Exponential", "params": {"loc": loc, "scale": scale}, "loglik": loglik, "k": 1}


def run_benchmark_competitions(seed=6000):
    """Fit TVMN/NUVM and standard benchmark distributions to raw positive datasets."""
    comparison_rows = []
    parameter_rows = []

    for dataset_index, (name, data) in enumerate(load_benchmark_datasets(), start=1):
        print(f"Benchmark data: {name}, n={len(data)}")
        fits = [
            fit_normal(data),
            fit_laplace(data),
            fit_student_t(data),
            fit_gamma(data),
            fit_lognormal(data),
            fit_weibull(data),
            fit_exponential(data),
            fit_nuvm(data, seed=seed + dataset_index),
            fit_tvmn_location(data, seed=seed + 100 + dataset_index),
        ]

        for fit in fits:
            pdf_func, cdf_func, _ = model_pdf_cdf_sf(fit)
            criteria = information_criteria(fit["loglik"], fit["k"], len(data))
            comparison_rows.append(
                {
                    "dataset": name,
                    "n": len(data),
                    "model": fit["model"],
                    "loglik": fit["loglik"],
                    "k": fit["k"],
                    "KS": empirical_ks(data, cdf_func),
                    **criteria,
                }
            )
            parameter_rows.append({"dataset": name, "model": fit["model"], **fit["params"]})

        plot_benchmark_fit(name, data, fits)

    comparison = pd.DataFrame(comparison_rows)
    parameters = pd.DataFrame(parameter_rows)
    comparison.to_csv(OUTPUT_DIR / "05_TVMN_benchmark_model_comparison.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "05_TVMN_benchmark_parameter_estimates.csv", index=False)
    summarize_benchmark_evidence(comparison)
    return comparison, parameters


def summarize_benchmark_evidence(comparison):
    """Write benchmark winner tables and a paper-facing interpretation."""
    rows = []
    tvmn_nuvm_rows = []
    for dataset, group in comparison.groupby("dataset", sort=False):
        group = group.copy()
        n = int(group["n"].iloc[0])
        for criterion in ["AIC", "BIC", "CAIC", "HQIC"]:
            best = group.loc[group[criterion].idxmin()]
            tvmn_value = float(group.loc[group["model"] == "TVMN", criterion].iloc[0])
            rows.append(
                {
                    "dataset": dataset,
                    "n": n,
                    "criterion": criterion,
                    "winner": best["model"],
                    "winner_value": float(best[criterion]),
                    "tvmn_value": tvmn_value,
                    "delta_tvmn_minus_best": tvmn_value - float(best[criterion]),
                }
            )
        tvmn = group.loc[group["model"] == "TVMN"].iloc[0]
        nuvm = group.loc[group["model"] == "NUVM"].iloc[0]
        tvmn_nuvm_rows.append(
            {
                "dataset": dataset,
                "n": n,
                "tvmn_loglik": float(tvmn["loglik"]),
                "nuvm_loglik": float(nuvm["loglik"]),
                "loglik_tvmn_minus_nuvm": float(tvmn["loglik"] - nuvm["loglik"]),
                "delta_aic_tvmn_minus_nuvm": float(tvmn["AIC"] - nuvm["AIC"]),
                "delta_bic_tvmn_minus_nuvm": float(tvmn["BIC"] - nuvm["BIC"]),
                "tvmn_ks": float(tvmn["KS"]),
                "nuvm_ks": float(nuvm["KS"]),
                "delta_ks_tvmn_minus_nuvm": float(tvmn["KS"] - nuvm["KS"]),
            }
        )

    winners = pd.DataFrame(rows)
    tvmn_nuvm = pd.DataFrame(tvmn_nuvm_rows)
    winners.to_csv(OUTPUT_DIR / "05_TVMN_benchmark_winners.csv", index=False)
    tvmn_nuvm.to_csv(OUTPUT_DIR / "05_TVMN_benchmark_TVMN_vs_NUVM.csv", index=False)

    aic_winners = winners[winners["criterion"] == "AIC"]
    counts = aic_winners["winner"].value_counts().sort_index()
    tvmn_wins = int(counts.get("TVMN", 0))
    tvmn_beats_nuvm_loglik = int((tvmn_nuvm["loglik_tvmn_minus_nuvm"] > 0).sum())
    tvmn_beats_nuvm_ks = int((tvmn_nuvm["delta_ks_tvmn_minus_nuvm"] < 0).sum())

    lines = [
        "# TVMN Strong Benchmark Evidence",
        "",
        "This section compares TVMN and NUVM against Normal, Laplace, Student-t, Gamma, "
        "Lognormal, Weibull, and Exponential models on raw positive datasets.",
        "",
        "## AIC Winners",
        "",
        "| Dataset | n | AIC winner | TVMN AIC - best AIC |",
        "|---|---:|---|---:|",
    ]
    for _, row in aic_winners.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n'])} | {row['winner']} | "
            f"{float(row['delta_tvmn_minus_best']):.3f} |"
        )

    lines.extend(
        [
            "",
            f"TVMN AIC wins: {tvmn_wins}/{len(aic_winners)}.",
            "",
            "## TVMN Versus NUVM",
            "",
            f"TVMN has higher log-likelihood than NUVM in {tvmn_beats_nuvm_loglik}/{len(tvmn_nuvm)} "
            "benchmark datasets.",
            "",
            f"TVMN has a smaller KS statistic than NUVM in {tvmn_beats_nuvm_ks}/{len(tvmn_nuvm)} "
            "benchmark datasets.",
            "",
            "The detailed comparison is saved in `05_TVMN_benchmark_TVMN_vs_NUVM.csv`.",
            "",
            "## Interpretation",
            "",
            "These expanded comparisons are the main reviewer-facing evidence. If TVMN wins or "
            "nearly ties on some datasets, the manuscript can claim competitive empirical "
            "performance. If it does not win, the paper should emphasize mathematical novelty, "
            "valid estimation, and situations where variance-mixture flexibility is practically useful.",
            "",
            "## Recommended Wording",
            "",
            "> Across several benchmark datasets, TVMN was compared with common symmetric, "
            "heavy-tailed, and positive-support distributions using log-likelihood and information "
            "criteria. The results show where the additional triangular variance-mixture structure "
            "is empirically useful and where simpler alternatives remain preferable.",
            "",
        ]
    )
    (OUTPUT_DIR / "05_TVMN_benchmark_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def plot_benchmark_fit(name, data, fits):
    """Create benchmark histogram, CDF, and hazard-shape plots."""
    safe_name = name.replace(" ", "_").replace("/", "_")
    x_grid = np.linspace(max(1e-8, np.min(data) * 0.8), np.max(data) * 1.15, 900)
    sorted_data = np.sort(data)
    empirical = np.arange(1, len(data) + 1) / len(data)

    ordered = sorted(fits, key=lambda fit: information_criteria(fit["loglik"], fit["k"], len(data))["AIC"])
    shown = ordered[:5]
    if not any(fit["model"] == "TVMN" for fit in shown):
        shown = shown[:4] + [next(fit for fit in fits if fit["model"] == "TVMN")]

    plt.figure(figsize=(9, 5.4))
    plt.hist(data, bins=28, density=True, alpha=0.35, edgecolor="white", label="Data")
    for fit in shown:
        pdf_func, _, _ = model_pdf_cdf_sf(fit)
        plt.plot(x_grid, pdf_func(x_grid), linewidth=2.0, label=fit["model"])
    plt.title(f"Benchmark Fitted PDFs: {name}")
    plt.xlabel("Observed value")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"12_{safe_name}_benchmark_histogram_pdfs.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.4))
    plt.step(sorted_data, empirical, where="post", linewidth=2.0, label="Empirical CDF")
    for fit in shown:
        _, cdf_func, _ = model_pdf_cdf_sf(fit)
        plt.plot(x_grid, cdf_func(x_grid), linewidth=2.0, label=fit["model"])
    plt.title(f"Benchmark Fitted CDFs: {name}")
    plt.xlabel("Observed value")
    plt.ylabel("CDF")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"13_{safe_name}_benchmark_cdfs.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.4))
    for fit in shown:
        pdf_func, _, sf_func = model_pdf_cdf_sf(fit)
        hazard = pdf_func(x_grid) / np.maximum(sf_func(x_grid), 1e-10)
        finite = np.isfinite(hazard)
        if np.any(finite):
            cap = np.nanpercentile(hazard[finite], 98.0)
            plt.plot(x_grid, np.minimum(hazard, cap), linewidth=2.0, label=fit["model"])
    plt.title(f"Benchmark Hazard Shapes: {name}")
    plt.xlabel("Observed value")
    plt.ylabel("Hazard")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"14_{safe_name}_benchmark_hazards.png", dpi=300)
    plt.close()


def information_criteria(loglik, k, n):
    """AIC, BIC, CAIC, and HQIC."""
    return {
        "AIC": -2.0 * loglik + 2.0 * k,
        "BIC": -2.0 * loglik + k * np.log(n),
        "CAIC": -2.0 * loglik + k * (np.log(n) + 1.0),
        "HQIC": -2.0 * loglik + 2.0 * k * np.log(np.log(n)),
    }


def run_real_data_applications(seed=4000):
    """Fit Normal, Laplace, NUVM, and TVMN to local real datasets."""
    comparison_rows = []
    parameter_rows = []

    for dataset_index, (name, data) in enumerate(load_real_datasets(), start=1):
        print(f"Real data: {name}, n={len(data)}")
        fits = [
            fit_normal(data),
            fit_laplace(data),
            fit_nuvm(data, seed=seed + dataset_index),
            fit_tvmn_real(data, seed=seed + 100 + dataset_index),
        ]

        for fit in fits:
            pdf_func, cdf_func = model_pdf_cdf(fit)
            criteria = information_criteria(fit["loglik"], fit["k"], len(data))
            ks = empirical_ks(data, cdf_func)
            comparison_rows.append(
                {
                    "dataset": name,
                    "n": len(data),
                    "model": fit["model"],
                    "loglik": fit["loglik"],
                    "k": fit["k"],
                    "KS": ks,
                    **criteria,
                }
            )
            parameter_rows.append({"dataset": name, "model": fit["model"], **fit["params"]})

        plot_real_data_fit(name, data, fits)

    comparison = pd.DataFrame(comparison_rows)
    parameters = pd.DataFrame(parameter_rows)
    comparison.to_csv(OUTPUT_DIR / "04_TVMN_real_data_model_comparison.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "04_TVMN_real_data_parameter_estimates.csv", index=False)
    summarize_real_data_evidence(comparison)
    return comparison, parameters


def summarize_real_data_evidence(comparison, min_publishable_n=30):
    """
    Write reviewer-facing summaries of the real-data evidence.

    TVMN and NUVM are not nested in the fitted parameterizations used here.
    Therefore, the likelihood comparison against NUVM is descriptive only; a
    formal likelihood-ratio test would require a nested model relationship.
    """
    winners = []
    tvmn_vs_nuvm = []

    for dataset, group in comparison.groupby("dataset", sort=False):
        group = group.copy()
        n = int(group["n"].iloc[0])
        publishable = n >= min_publishable_n
        for criterion in ["AIC", "BIC", "CAIC", "HQIC"]:
            best = group.loc[group[criterion].idxmin()]
            tvmn_value = float(group.loc[group["model"] == "TVMN", criterion].iloc[0])
            winners.append(
                {
                    "dataset": dataset,
                    "n": n,
                    "publishable_n": publishable,
                    "criterion": criterion,
                    "winner": best["model"],
                    "winner_value": float(best[criterion]),
                    "tvmn_value": tvmn_value,
                    "delta_tvmn_minus_best": tvmn_value - float(best[criterion]),
                }
            )

        tvmn = group.loc[group["model"] == "TVMN"].iloc[0]
        nuvm = group.loc[group["model"] == "NUVM"].iloc[0]
        tvmn_vs_nuvm.append(
            {
                "dataset": dataset,
                "n": n,
                "publishable_n": publishable,
                "tvmn_loglik": float(tvmn["loglik"]),
                "nuvm_loglik": float(nuvm["loglik"]),
                "loglik_tvmn_minus_nuvm": float(tvmn["loglik"] - nuvm["loglik"]),
                "two_delta_loglik_descriptive": float(2.0 * (tvmn["loglik"] - nuvm["loglik"])),
                "delta_aic_tvmn_minus_nuvm": float(tvmn["AIC"] - nuvm["AIC"]),
                "delta_bic_tvmn_minus_nuvm": float(tvmn["BIC"] - nuvm["BIC"]),
                "note": "Descriptive only: TVMN and NUVM are non-nested in this implementation.",
            }
        )

    winners_df = pd.DataFrame(winners)
    tvmn_vs_nuvm_df = pd.DataFrame(tvmn_vs_nuvm)
    winners_df.to_csv(OUTPUT_DIR / "04_TVMN_real_data_winners.csv", index=False)
    tvmn_vs_nuvm_df.to_csv(OUTPUT_DIR / "04_TVMN_vs_NUVM_likelihood_comparison.csv", index=False)

    publishable_winners = winners_df[(winners_df["publishable_n"]) & (winners_df["criterion"] == "AIC")]
    winner_counts = publishable_winners["winner"].value_counts().sort_index()
    tvmn_wins = int(winner_counts.get("TVMN", 0))
    total_publishable = int(len(publishable_winners))

    lines = [
        "# TVMN Real-Data Evidence Interpretation",
        "",
        "## Main conclusion",
        "",
        "The real-data evidence should be presented as competitive rather than superior. "
        "In the current datasets with publishable sample sizes, TVMN does not attain the minimum AIC.",
        "",
        "A defensible claim is:",
        "",
        "> TVMN is a new flexible variance-mixture distribution with tractable properties, "
        "valid estimation procedures, and competitive performance on real datasets.",
        "",
        "Avoid claiming that TVMN is uniformly better than Normal, Laplace, or NUVM models.",
        "",
        "## AIC winners",
        "",
        "| Dataset | n | AIC winner | TVMN AIC - best AIC |",
        "|---|---:|---|---:|",
    ]

    for _, row in publishable_winners.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n'])} | {row['winner']} | "
            f"{float(row['delta_tvmn_minus_best']):.3f} |"
        )

    lines.extend(
        [
            "",
            f"TVMN AIC wins among publishable-size datasets: {tvmn_wins}/{total_publishable}.",
            "",
            "## Small-sample caution",
            "",
            f"Datasets with n < {min_publishable_n}, such as the Linnerud example, should be treated "
            "as diagnostic illustrations rather than substantive real-data evidence.",
            "",
            "## TVMN versus NUVM",
            "",
            "The file `04_TVMN_vs_NUVM_likelihood_comparison.csv` reports descriptive log-likelihood "
            "and information-criterion differences. A formal likelihood-ratio test is not reported "
            "because TVMN and NUVM are non-nested in the present parameterizations.",
            "",
            "## Publication strategy",
            "",
            "The paper is strongest as a theoretical and estimation contribution. To strengthen the "
            "empirical section, add 10-20 real datasets, especially heavy-tailed finance, insurance, "
            "rainfall, and reliability examples, and report cases where TVMN materially improves AIC, "
            "BIC, or tail-fit diagnostics.",
            "",
        ]
    )

    (OUTPUT_DIR / "04_TVMN_real_data_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def plot_real_data_fit(name, data, fits):
    """Create histogram/PDF, empirical CDF, and Q-Q style diagnostic plots."""
    safe_name = name.replace(" ", "_").replace("/", "_")
    x_grid = np.linspace(np.min(data) - 0.15 * np.std(data), np.max(data) + 0.15 * np.std(data), 900)
    sorted_data = np.sort(data)
    empirical = np.arange(1, len(data) + 1) / len(data)

    plt.figure(figsize=(9, 5.4))
    plt.hist(data, bins=28, density=True, alpha=0.35, edgecolor="white", label="Data")
    for fit in fits:
        pdf_func, _ = model_pdf_cdf(fit)
        plt.plot(x_grid, pdf_func(x_grid), linewidth=2.0, label=fit["model"])
    plt.title(f"Histogram with Fitted PDFs: {name}")
    plt.xlabel("Centered value")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"08_{safe_name}_histogram_fitted_pdfs.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.4))
    plt.step(sorted_data, empirical, where="post", linewidth=2.0, label="Empirical CDF")
    for fit in fits:
        _, cdf_func = model_pdf_cdf(fit)
        plt.plot(x_grid, cdf_func(x_grid), linewidth=2.0, label=fit["model"])
    plt.title(f"Empirical CDF with Fitted CDFs: {name}")
    plt.xlabel("Centered value")
    plt.ylabel("CDF")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"09_{safe_name}_empirical_cdf.png", dpi=300)
    plt.close()

    tvmn_fit = next(fit for fit in fits if fit["model"] == "TVMN")
    _, tvmn_cdf_func = model_pdf_cdf(tvmn_fit)
    probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    q_min, q_max = np.min(sorted_data), np.max(sorted_data)
    q_grid = np.linspace(q_min, q_max, 2000)
    fitted_cdf_grid = tvmn_cdf_func(q_grid)
    theoretical_quantiles = np.interp(probs, fitted_cdf_grid, q_grid)

    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(theoretical_quantiles, sorted_data, s=18, alpha=0.75)
    lo = min(np.min(theoretical_quantiles), np.min(sorted_data))
    hi = max(np.max(theoretical_quantiles), np.max(sorted_data))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.4)
    plt.title(f"TVMN Q-Q Plot: {name}")
    plt.xlabel("TVMN theoretical quantiles")
    plt.ylabel("Empirical quantiles")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"10_{safe_name}_TVMN_QQ_plot.png", dpi=300)
    plt.close()


def run_bootstrap(n=250, true_params=(0.5, 1.0, 2.0), replications=200, seed=9090):
    """Parametric bootstrap confidence intervals for TVMN MLE."""
    rng = np.random.default_rng(seed)
    base_data = tvmn_mle.generate_tvmn_sample(n, *true_params, seed=seed)
    params_hat, fit = estimate_tvmn(base_data, n_starts=24, seed=seed + 1)

    rows = []
    for rep in range(1, replications + 1):
        sample_seed = int(rng.integers(1, 2_000_000_000))
        boot_data = tvmn_mle.generate_tvmn_sample(n, *params_hat, seed=sample_seed)
        boot_fit = tvmn_mle.estimate_tvmn_mle(boot_data, n_starts=8, seed=sample_seed + 1)
        rows.append(
            {
                "replication": rep,
                "success": boot_fit["success"],
                "a_hat": boot_fit["a_hat"],
                "m_hat": boot_fit["m_hat"],
                "b_hat": boot_fit["b_hat"],
                "loglik": boot_fit["loglik"],
            }
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "04_TVMN_bootstrap_raw.csv", index=False)

    successful = raw[raw["success"]].copy()
    ci_rows = []
    for name, base_estimate in zip(["a", "m", "b"], params_hat):
        estimates = successful[f"{name}_hat"].to_numpy(dtype=float)
        ci_rows.append(
            {
                "parameter": name,
                "base_estimate": base_estimate,
                "bootstrap_mean": float(np.mean(estimates)) if len(estimates) else np.nan,
                "bootstrap_sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else np.nan,
                "ci_lower_percentile": float(np.percentile(estimates, 2.5)) if len(estimates) else np.nan,
                "ci_upper_percentile": float(np.percentile(estimates, 97.5)) if len(estimates) else np.nan,
                "successful_bootstrap_replications": len(successful),
            }
        )

    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUTPUT_DIR / "04_TVMN_bootstrap_confidence_intervals.csv", index=False)
    plot_bootstrap_distributions(successful, params_hat)
    return raw, ci, fit


def plot_bootstrap_distributions(successful, params_hat):
    """Plot bootstrap distributions of a, m, and b estimates."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name, estimate in zip(axes, ["a", "m", "b"], params_hat):
        ax.hist(successful[f"{name}_hat"], bins=30, alpha=0.7, edgecolor="white")
        ax.axvline(estimate, linestyle="--", color="black", linewidth=1.5)
        ax.set_title(f"Bootstrap {name}_hat")
        ax.set_xlabel(name)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_TVMN_bootstrap_distributions.png", dpi=300)
    plt.close()


def write_master_notes(args):
    """Write a compact notes file describing generated empirical outputs."""
    notes_path = OUTPUT_DIR / "04_TVMN_empirical_studies_notes.md"
    with notes_path.open("w", encoding="utf-8") as handle:
        handle.write("# TVMN Empirical Studies Notes\n\n")
        handle.write("This script covers Fisher information, Monte Carlo study, real-data applications, and bootstrap.\n\n")
        handle.write("## Recommended Paper-Scale Runs\n\n")
        handle.write("```powershell\n")
        handle.write("python TVMN\\04_TVMN_empirical_studies.py --part monte-carlo --mc-replications 1000\n")
        handle.write("python TVMN\\04_TVMN_empirical_studies.py --part bootstrap --bootstrap-replications 1000\n")
        handle.write("python TVMN\\04_TVMN_empirical_studies.py --part real-data\n")
        handle.write("```\n\n")
        handle.write("## Current Run Settings\n\n")
        handle.write(f"- part: {args.part}\n")
        handle.write(f"- Monte Carlo replications: {args.mc_replications}\n")
        handle.write(f"- bootstrap replications: {args.bootstrap_replications}\n")
        handle.write(f"- bootstrap n: {args.bootstrap_n}\n")
        handle.write(f"- Fisher n: {args.fisher_n}\n")
        handle.write("\n")
        handle.write("## Interpretation Outputs\n\n")
        handle.write("- `04_TVMN_real_data_interpretation.md`: reviewer-facing summary of the mixed real-data evidence.\n")
        handle.write("- `04_TVMN_real_data_winners.csv`: information-criterion winners and TVMN deltas.\n")
        handle.write("- `04_TVMN_vs_NUVM_likelihood_comparison.csv`: descriptive TVMN-versus-NUVM likelihood differences.\n")
        handle.write("- `05_TVMN_benchmark_model_comparison.csv`: expanded benchmark comparison against 8 competitors.\n")
        handle.write("- `05_TVMN_benchmark_interpretation.md`: reviewer-facing benchmark interpretation.\n")
        handle.write("\n")
        handle.write("## Publication Framing\n\n")
        handle.write(
            "Use a competitive-performance claim for the current manuscript. The real-data evidence "
            "does not yet support a universal superiority claim for TVMN.\n"
        )
        handle.write(
            "\nThe existing Monte Carlo and bootstrap CSV files may come from larger paper-scale runs "
            "than the settings shown above if only one section was regenerated.\n"
        )
    return notes_path


def main():
    parser = argparse.ArgumentParser(description="TVMN Fisher information, Monte Carlo, real data, and bootstrap.")
    parser.add_argument(
        "--part",
        choices=["all", "fisher", "monte-carlo", "real-data", "benchmark", "bootstrap"],
        default="all",
    )
    parser.add_argument("--mc-replications", type=int, default=50)
    parser.add_argument("--bootstrap-replications", type=int, default=200)
    parser.add_argument("--bootstrap-n", type=int, default=250)
    parser.add_argument("--fisher-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ensure_output_dirs()

    if args.part in ["all", "fisher"]:
        print("Running Fisher information section...")
        run_fisher_section(n=args.fisher_n, seed=args.seed)

    if args.part in ["all", "monte-carlo"]:
        print("Running Monte Carlo section...")
        run_monte_carlo(replications=args.mc_replications, seed=args.seed)

    if args.part in ["all", "real-data"]:
        print("Running real-data applications...")
        run_real_data_applications(seed=args.seed)

    if args.part in ["all", "benchmark"]:
        print("Running expanded benchmark competitions...")
        run_benchmark_competitions(seed=args.seed)

    if args.part in ["all", "bootstrap"]:
        print("Running bootstrap section...")
        run_bootstrap(n=args.bootstrap_n, replications=args.bootstrap_replications, seed=args.seed)

    notes_path = write_master_notes(args)
    print("Done.")
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()
