"""
TVMN Likelihood and MLE Recovery

This script starts the estimation stage only.

Completed here:
    1. Define likelihood L(a,m,b).
    2. Define log-likelihood ell(a,m,b).
    3. Implement neg_loglik(params, data) with a > 0 and a < m < b.
    4. Generate artificial TVMN data with true parameters a=0.5, m=1.0, b=2.0.
    5. Estimate a, m, b by scipy.optimize.minimize.
    6. Check parameter recovery for n=100, n=200, and n=500.

Not included here:
    Fisher information, standard errors, confidence intervals,
    Monte Carlo simulation study, real data, or bootstrap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_module(module_name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tvmn_core = load_module("tvmn_core", BASE_DIR / "01_TVMN_step_by_step_derivation.py")


def ensure_output_dirs():
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def generate_tvmn_sample(n, a, m, b, seed=123):
    """
    Generate X from the TVMN model.

    Algorithm:
        V ~ Triangular(a, m, b)
        X | V ~ Normal(0, V)
    """
    rng = np.random.default_rng(seed)
    v = rng.triangular(left=a, mode=m, right=b, size=n)
    x = rng.normal(loc=0.0, scale=np.sqrt(v), size=n)
    return x


def loglik(params, data):
    """
    TVMN log-likelihood.

    For observations x_1,...,x_n:

        L(a,m,b) = product_{i=1}^n f(x_i; a,m,b)

    and

        ell(a,m,b) = sum_{i=1}^n log f(x_i; a,m,b).

    Constraints:
        a > 0
        a < m < b
    """
    a, m, b = np.asarray(params, dtype=float)
    data = np.asarray(data, dtype=float)

    if not np.isfinite(a + m + b):
        return -np.inf
    if not (a > 0.0 and a < m < b):
        return -np.inf

    pdf_values = tvmn_core.tvmn_pdf(data, a, m, b)
    if np.any(~np.isfinite(pdf_values)) or np.any(pdf_values <= 0.0):
        return -np.inf

    value = np.sum(np.log(np.maximum(pdf_values, 1e-300)))
    if not np.isfinite(value):
        return -np.inf

    return float(value)


def neg_loglik(params, data):
    """Negative log-likelihood in native parameters (a,m,b)."""
    value = loglik(params, data)
    if not np.isfinite(value):
        return 1e100
    return -value


def params_to_theta(params):
    """
    Convert native parameters to unconstrained positive-gap parameters.

        a       = exp(theta_0)
        m - a   = exp(theta_1)
        b - m   = exp(theta_2)
    """
    a, m, b = np.asarray(params, dtype=float)
    return np.log([a, m - a, b - m])


def theta_to_params(theta):
    """Convert positive-gap theta values back to native TVMN parameters."""
    gap_a, gap_m, gap_b = np.exp(np.asarray(theta, dtype=float))
    a = gap_a
    m = a + gap_m
    b = m + gap_b
    return np.array([a, m, b], dtype=float)


def neg_loglik_theta(theta, data, upper_b):
    """
    Negative log-likelihood in transformed parameters.

    The transformation enforces a > 0 and a < m < b.
    A broad upper bound on b avoids numerically meaningless variance ranges
    during optimization.
    """
    params = theta_to_params(theta)
    if params[2] > upper_b:
        return 1e100 + (params[2] - upper_b) ** 2
    return neg_loglik(params, data)


def moment_based_starts(data):
    """
    Build several reasonable starting values from the sample variance.

    Since Var(X)=E[V]=(a+m+b)/3, these starts are scaled so their average
    variance is close to the sample variance.
    """
    data = np.asarray(data, dtype=float)
    sample_variance = max(float(np.var(data, ddof=0)), 1e-4)

    prototype_shapes = [
        (0.5, 1.0, 2.0),
        (0.2, 1.0, 1.8),
        (0.8, 1.0, 1.2),
        (0.5, 1.5, 3.0),
        (1.0, 2.0, 5.0),
        (0.3, 0.8, 2.5),
        (0.7, 1.8, 2.2),
    ]

    starts = []
    for shape in prototype_shapes:
        shape = np.array(shape, dtype=float)
        scale = sample_variance / np.mean(shape)
        starts.append(shape * scale)

    lower = max(0.05 * sample_variance, 1e-4)
    center = sample_variance
    upper = max(2.5 * sample_variance, lower + 2e-4)
    starts.append(np.array([lower, center, upper], dtype=float))

    return starts


def estimate_tvmn_mle(data, n_starts=18, seed=123, verbose=False):
    """
    Estimate TVMN parameters by transformed L-BFGS-B multi-start MLE.

    Returns a dictionary with estimates, log-likelihood, and optimizer status.
    """
    data = np.asarray(data, dtype=float)
    sample_variance = max(float(np.var(data, ddof=0)), 1e-4)
    max_square = max(float(np.max(data * data)), sample_variance)
    upper_b = max(20.0 * sample_variance, 4.0 * max_square, 5.0)
    min_gap = max(1e-5, 1e-6 * sample_variance)

    theta_bounds = [
        (np.log(min_gap), np.log(upper_b)),
        (np.log(min_gap), np.log(upper_b)),
        (np.log(min_gap), np.log(upper_b)),
    ]

    starts = moment_based_starts(data)
    rng = np.random.default_rng(seed)

    while len(starts) < n_starts:
        raw = np.sort(rng.uniform(0.05 * sample_variance, 3.0 * sample_variance, size=3))
        if raw[0] > 0 and raw[0] < raw[1] < raw[2]:
            starts.append(raw)

    best_result = None
    best_params = None
    best_nll = np.inf

    for index, start_params in enumerate(starts[:n_starts]):
        if not (start_params[0] > 0 and start_params[0] < start_params[1] < start_params[2]):
            continue

        theta0 = params_to_theta(start_params)
        result = minimize(
            neg_loglik_theta,
            x0=theta0,
            args=(data, upper_b),
            method="L-BFGS-B",
            bounds=theta_bounds,
            options={
                "maxiter": 2500,
                "ftol": 1e-11,
                "gtol": 1e-6,
                "maxls": 50,
            },
        )

        params_hat = theta_to_params(result.x)
        current_nll = neg_loglik(params_hat, data)
        valid = np.isfinite(current_nll) and params_hat[0] > 0 and params_hat[0] < params_hat[1] < params_hat[2]

        if verbose:
            print(f"start {index + 1}: success={result.success}, nll={current_nll:.6f}, params={params_hat}")

        if valid and current_nll < best_nll:
            best_nll = current_nll
            best_result = result
            best_params = params_hat

    if best_result is None:
        return {
            "success": False,
            "a_hat": np.nan,
            "m_hat": np.nan,
            "b_hat": np.nan,
            "loglik": -np.inf,
            "nll": np.inf,
            "message": "All optimization attempts failed.",
        }

    return {
        "success": bool(best_result.success or np.isfinite(best_nll)),
        "a_hat": float(best_params[0]),
        "m_hat": float(best_params[1]),
        "b_hat": float(best_params[2]),
        "loglik": float(-best_nll),
        "nll": float(best_nll),
        "message": str(best_result.message),
        "raw_result": best_result,
    }


def plot_recovery(results, true_params):
    """Create a simple parameter recovery plot across sample sizes."""
    n_values = [row["n"] for row in results]
    names = ["a", "m", "b"]
    estimate_keys = ["a_hat", "m_hat", "b_hat"]

    plt.figure(figsize=(8.5, 5.2))
    for idx, name in enumerate(names):
        estimates = [row[estimate_keys[idx]] for row in results]
        plt.plot(n_values, estimates, marker="o", linewidth=2.0, label=f"{name}_hat")
        plt.axhline(true_params[idx], linestyle="--", linewidth=1.5, alpha=0.75, label=f"true {name}")

    plt.title("TVMN MLE Parameter Recovery")
    plt.xlabel("Sample size")
    plt.ylabel("Parameter value")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()

    output_path = FIGURE_DIR / "06_TVMN_MLE_parameter_recovery.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def write_results(results, true_params):
    """Save MLE recovery results and likelihood notes."""
    csv_path = OUTPUT_DIR / "03_TVMN_MLE_recovery.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "n,true_a,true_m,true_b,a_hat,m_hat,b_hat,"
            "m_minus_a,b_minus_m,bias_a,bias_m,bias_b,"
            "loglik_hat,loglik_true,loglik_gain,nll,near_degenerate,success,message\n"
        )
        for row in results:
            gap_left = row["m_hat"] - row["a_hat"]
            gap_right = row["b_hat"] - row["m_hat"]
            near_degenerate = gap_left < 0.10 or gap_right < 0.10
            handle.write(
                f"{row['n']},{true_params[0]},{true_params[1]},{true_params[2]},"
                f"{row['a_hat']},{row['m_hat']},{row['b_hat']},"
                f"{gap_left},{gap_right},"
                f"{row['a_hat'] - true_params[0]},{row['m_hat'] - true_params[1]},{row['b_hat'] - true_params[2]},"
                f"{row['loglik']},{row['loglik_true']},{row['loglik'] - row['loglik_true']},"
                f"{row['nll']},{near_degenerate},{row['success']},{row['message']}\n"
            )

    notes_path = OUTPUT_DIR / "03_TVMN_likelihood_mle_notes.md"
    with notes_path.open("w", encoding="utf-8") as handle:
        handle.write("# TVMN Likelihood and MLE Notes\n\n")
        handle.write("This stage defines and tests likelihood-based estimation only.\n\n")
        handle.write("## Likelihood\n\n")
        handle.write("For independent observations `x_1,...,x_n`,\n\n")
        handle.write("```text\n")
        handle.write("L(a,m,b) = product_{i=1}^n f(x_i; a,m,b)\n")
        handle.write("```\n\n")
        handle.write("where `f(x_i; a,m,b)` is the TVMN PDF derived in Stage 1.\n\n")
        handle.write("## Log-likelihood\n\n")
        handle.write("```text\n")
        handle.write("ell(a,m,b) = sum_{i=1}^n log f(x_i; a,m,b)\n")
        handle.write("```\n\n")
        handle.write("The numerical optimization minimizes the negative log-likelihood.\n\n")
        handle.write("## Constraints\n\n")
        handle.write("```text\n")
        handle.write("a > 0\n")
        handle.write("a < m < b\n")
        handle.write("```\n\n")
        handle.write("Internally, the optimizer uses the positive-gap transformation:\n\n")
        handle.write("```text\n")
        handle.write("a = exp(theta_0)\n")
        handle.write("m = a + exp(theta_1)\n")
        handle.write("b = m + exp(theta_2)\n")
        handle.write("```\n\n")
        handle.write("This automatically preserves `a > 0` and `a < m < b`.\n\n")
        handle.write("## Artificial Data Recovery\n\n")
        handle.write(f"True parameters: `a={true_params[0]}`, `m={true_params[1]}`, `b={true_params[2]}`.\n\n")
        handle.write("| n | a_hat | m_hat | b_hat | loglik |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row['n']} | {row['a_hat']:.6f} | {row['m_hat']:.6f} | "
                f"{row['b_hat']:.6f} | {row['loglik']:.6f} |\n"
            )
        handle.write("\n")
        handle.write("## Recovery Diagnostic\n\n")
        handle.write("The table below compares the log-likelihood at the MLE with the log-likelihood at the true data-generating parameters.\n\n")
        handle.write("| n | loglik_hat | loglik_true | gain | m_hat-a_hat | b_hat-m_hat |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            gap_left = row["m_hat"] - row["a_hat"]
            gap_right = row["b_hat"] - row["m_hat"]
            handle.write(
                f"| {row['n']} | {row['loglik']:.6f} | {row['loglik_true']:.6f} | "
                f"{row['loglik'] - row['loglik_true']:.6f} | {gap_left:.6f} | {gap_right:.6f} |\n"
            )
        handle.write("\n")
        handle.write(
            "For smaller samples, the likelihood may prefer a very narrow triangular variance interval. "
            "This is a useful warning for the next stage: the finite-sample behavior of the three TVMN "
            "variance parameters must be studied carefully by Monte Carlo simulation.\n\n"
        )
        handle.write("These runs are a first recovery check. The formal Monte Carlo bias, MSE, and RMSE study belongs to the next stage.\n")

    return csv_path, notes_path


def print_likelihood_section():
    """Print the estimation-theory formulas in the terminal."""
    print("TVMN Stage 5: Likelihood and MLE")
    print("=" * 70)
    print("For data x_1,...,x_n:")
    print("L(a,m,b) = product_{i=1}^n f(x_i; a,m,b)")
    print("ell(a,m,b) = sum_{i=1}^n log f(x_i; a,m,b)")
    print("MLE: maximize ell(a,m,b), equivalently minimize -ell(a,m,b).")
    print("Constraints: a > 0 and a < m < b.")
    print("Internal optimizer transformation:")
    print("a = exp(theta_0)")
    print("m = a + exp(theta_1)")
    print("b = m + exp(theta_2)")
    print()


def main():
    ensure_output_dirs()
    print_likelihood_section()

    true_params = (0.5, 1.0, 2.0)
    sample_sizes = [100, 200, 500]
    results = []

    for n in sample_sizes:
        data = generate_tvmn_sample(n, *true_params, seed=2026 + n)
        fit = estimate_tvmn_mle(data, n_starts=24, seed=9000 + n, verbose=False)
        loglik_true = loglik(true_params, data)
        row = {
            "n": n,
            "sample_mean": float(np.mean(data)),
            "sample_variance": float(np.var(data, ddof=1)),
            "loglik_true": float(loglik_true),
            **fit,
        }
        results.append(row)

        print(f"n = {n}")
        print(f"  sample mean     = {row['sample_mean']:.6f}")
        print(f"  sample variance = {row['sample_variance']:.6f}")
        print(f"  true params     = a={true_params[0]:.6f}, m={true_params[1]:.6f}, b={true_params[2]:.6f}")
        print(f"  estimates       = a={row['a_hat']:.6f}, m={row['m_hat']:.6f}, b={row['b_hat']:.6f}")
        print(f"  loglik          = {row['loglik']:.6f}")
        print(f"  loglik at true  = {row['loglik_true']:.6f}")
        print(f"  loglik gain     = {row['loglik'] - row['loglik_true']:.6f}")
        print(f"  gaps            = m-a={row['m_hat'] - row['a_hat']:.6f}, b-m={row['b_hat'] - row['m_hat']:.6f}")
        print(f"  success         = {row['success']}")
        print()

    plot_path = plot_recovery(results, true_params)
    csv_path, notes_path = write_results(results, true_params)

    print("Saved outputs:")
    print(csv_path)
    print(notes_path)
    print(plot_path)


if __name__ == "__main__":
    main()
