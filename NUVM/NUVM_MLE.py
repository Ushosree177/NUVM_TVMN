import numpy as np
import pandas as pd
import argparse
from scipy.optimize import minimize
from scipy.special import erf


def nuvm_pdf(x, mu, sigma, a):
    """
    Numerically stable marginal PDF for the NUVM model:
        V ~ Uniform((1-a)sigma^2, (1+a)sigma^2)
        X | V ~ N(mu, V)

    Parameters
    ----------
    x : array-like
    mu : float
    sigma : float, must satisfy sigma > 0
    a : float, must satisfy 0 < a < 1

    Returns
    -------
    ndarray or float
    """
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)

    if sigma <= 0 or a <= 0 or a >= 1:
        out = np.full(x.shape, 1e-300)
        return float(out[0]) if scalar_input else out

    z = x - mu
    abs_z = np.abs(z)

    lower_v = (1.0 - a) * sigma * sigma
    upper_v = (1.0 + a) * sigma * sigma
    width_v = 2.0 * a * sigma * sigma

    def antiderivative(v, az):
        safe_v = np.maximum(v, 1e-300)
        expo_arg = np.minimum((az * az) / (2.0 * safe_v), 700.0)
        term_1 = 2.0 * np.sqrt(safe_v) * np.exp(-expo_arg)
        term_2 = np.sqrt(2.0 * np.pi) * az * erf(az / np.sqrt(2.0 * safe_v))
        return term_1 + term_2

    density_at_zero = 2.0 * (np.sqrt(upper_v) - np.sqrt(lower_v)) / (width_v * np.sqrt(2.0 * np.pi))

    with np.errstate(all='ignore'):
        density_nonzero = (antiderivative(upper_v, abs_z) - antiderivative(lower_v, abs_z)) / (width_v * np.sqrt(2.0 * np.pi))

    density = np.where(abs_z < 1e-12, density_at_zero, density_nonzero)
    density = np.maximum(density, 1e-300)

    return float(density[0]) if scalar_input else density


def log_likelihood_nuvm(params, data):
    """
    Log-likelihood for NUVM using native parameters:
        params = [mu, sigma, a]
    """
    mu = params[0]
    sigma = params[1]
    a = params[2]

    if sigma <= 0 or a <= 0 or a >= 1:
        return -np.inf

    pdf_vals = nuvm_pdf(data, mu, sigma, a)
    if np.any(~np.isfinite(pdf_vals)) or np.any(pdf_vals <= 0):
        return -np.inf

    ll_val = np.sum(np.log(pdf_vals))

    if not np.isfinite(ll_val):
        return -np.inf

    return float(ll_val)


def neg_log_likelihood_nuvm(params, data):
    ll_val = log_likelihood_nuvm(params, data)
    if not np.isfinite(ll_val):
        return 1e100
    return -ll_val


def mom_start_nuvm(data):
    """
    Method-of-moments style starting values.
    Useful as a stable warm start for MLE.
    """
    data = np.asarray(data, dtype=float)

    mu_0 = np.mean(data)
    sigma_0 = np.std(data, ddof=1)
    sigma_0 = max(float(sigma_0), 1e-3)

    centered = data - mu_0
    variance_0 = np.mean(centered ** 2)

    if variance_0 <= 0:
        a_0 = 0.3
    else:
        fourth_moment = np.mean(centered ** 4)
        raw_kurtosis = fourth_moment / (variance_0 ** 2)
        excess_kurtosis = max(raw_kurtosis - 3.0, 0.0)
        a_0 = np.sqrt(excess_kurtosis)

    a_0 = float(np.clip(a_0, 0.02, 0.95))

    return np.array([mu_0, sigma_0, a_0], dtype=float)


def fit_nuvm_mle(data, n_starts=8, seed=123, verbose=False):
    """
    Full fixed MLE for NUVM:
    - L-BFGS-B
    - direct parameterization
    - parameter bounds
    - multi-start optimization
    """
    data = np.asarray(data, dtype=float)

    bounds = [
        (None, None),     # mu
        (1e-4, None),     # sigma > 0
        (1e-4, 0.999)     # 0 < a < 1
    ]

    start_main = mom_start_nuvm(data)

    rng_obj = np.random.default_rng(seed)
    starting_points = [start_main.copy()]

    for _ in range(n_starts - 1):
        mu_s = start_main[0] + rng_obj.normal(0.0, max(0.1, 0.1 * start_main[1]))
        sigma_s = start_main[1] * rng_obj.uniform(0.7, 1.3)
        sigma_s = max(sigma_s, 1e-3)
        a_s = rng_obj.uniform(0.05, 0.90)
        starting_points.append(np.array([mu_s, sigma_s, a_s], dtype=float))

    best_result = None
    best_nll = np.inf

    for idx, theta_0 in enumerate(starting_points):
        try:
            opt_res = minimize(
                neg_log_likelihood_nuvm,
                x0=theta_0,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-13,
                    'gtol': 1e-8,
                    'eps': 1e-7
                }
            )

            mu_hat = opt_res.x[0]
            sigma_hat = opt_res.x[1]
            a_hat = opt_res.x[2]

            valid = np.isfinite(opt_res.fun) and sigma_hat > 0 and 0 < a_hat < 1

            if verbose:
                print('start', idx + 1)
                print(opt_res.success)
                print(opt_res.fun)
                print(opt_res.x)
                print(valid)

            if valid and opt_res.fun < best_nll:
                best_nll = opt_res.fun
                best_result = opt_res

        except Exception:
            continue

    if best_result is None:
        return {
            'success': False,
            'mu': np.nan,
            'sigma': np.nan,
            'a': np.nan,
            'loglik': -np.inf,
            'nll': np.inf,
            'message': 'All optimization attempts failed'
        }

    return {
        'success': bool(np.isfinite(best_result.fun) and best_result.x[1] > 0 and 0 < best_result.x[2] < 1),
        'mu': float(best_result.x[0]),
        'sigma': float(best_result.x[1]),
        'a': float(best_result.x[2]),
        'loglik': float(-best_result.fun),
        'nll': float(best_result.fun),
        'message': best_result.message,
        'raw_result': best_result
    }


def generate_nuvm(n, mu, sigma, a, seed=123):
    """Generate a random NUVM sample for testing the MLE file directly."""
    rng = np.random.default_rng(seed)
    lower_v = (1.0 - a) * sigma * sigma
    upper_v = (1.0 + a) * sigma * sigma
    v = rng.uniform(lower_v, upper_v, size=n)
    return rng.normal(mu, np.sqrt(v))


def load_data_from_csv(path, column=None):
    """Load one numeric column from a CSV file."""
    df = pd.read_csv(path)

    if column is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in the CSV file.")
        column = numeric_columns[0]

    data = df[column].dropna().to_numpy(dtype=float)
    if len(data) == 0:
        raise ValueError("Selected column has no numeric observations.")
    return data, column


def print_mle_result(result):
    """Print a clean MLE result table."""
    print("\nNUVM MLE RESULTS")
    print("----------------")
    print(f"success : {result['success']}")
    print(f"mu      : {result['mu']:.10f}")
    print(f"sigma   : {result['sigma']:.10f}")
    print(f"a       : {result['a']:.10f}")
    print(f"loglik  : {result['loglik']:.10f}")
    print(f"nll     : {result['nll']:.10f}")
    print(f"message : {result['message']}")


def main():
    parser = argparse.ArgumentParser(description="Fit the NUVM distribution by corrected bounded MLE.")
    parser.add_argument("--csv", default=None, help="Optional CSV file containing data.")
    parser.add_argument("--column", default=None, help="Column name to use from the CSV file.")
    parser.add_argument("--n", type=int, default=500, help="Sample size for generated test data.")
    parser.add_argument("--true-mu", type=float, default=0.0, help="True mu for generated test data.")
    parser.add_argument("--true-sigma", type=float, default=1.0, help="True sigma for generated test data.")
    parser.add_argument("--true-a", type=float, default=0.5, help="True a for generated test data.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--n-starts", type=int, default=8, help="Number of optimization starts.")
    args = parser.parse_args()

    if args.csv:
        data, column = load_data_from_csv(args.csv, args.column)
        print(f"Loaded {len(data)} observations from column: {column}")
    else:
        data = generate_nuvm(
            n=args.n,
            mu=args.true_mu,
            sigma=args.true_sigma,
            a=args.true_a,
            seed=args.seed,
        )
        print("Generated NUVM test data")
        print(f"true mu    : {args.true_mu}")
        print(f"true sigma : {args.true_sigma}")
        print(f"true a     : {args.true_a}")
        print(f"n          : {args.n}")

    result = fit_nuvm_mle(data, n_starts=args.n_starts, seed=args.seed)
    print_mle_result(result)


if __name__ == "__main__":
    main()
