"""
Reviewer-strengthening additions for the TVMN/NUVM manuscript.

This script adds the experiments and figures most likely to answer reviewer
concerns about finite-sample identifiability and applied breadth:

    1. Large-sample TVMN recovery at n = 1000, 2000, and 5000.
    2. Likelihood contour plots for (a, m) and (m, b).
    3. Uniform-vs-triangular variance mixing-law comparison figure.
    4. A reliability/failure-time benchmark based on the Aarset data.

Run from the project root:

    python TVMN\\06_TVMN_reviewer_strengthening.py --part all

For a faster dry run:

    python TVMN\\06_TVMN_reviewer_strengthening.py --part all --recovery-sizes 1000
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


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FIGURE_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tvmn_core = load_module("tvmn_core_reviewer", BASE_DIR / "01_TVMN_step_by_step_derivation.py")
tvmn_mle = load_module("tvmn_mle_reviewer", BASE_DIR / "03_TVMN_likelihood_mle.py")
empirical = load_module("tvmn_empirical_reviewer", BASE_DIR / "04_TVMN_empirical_studies.py")


def ensure_output_dirs():
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def aarset_failure_times():
    """
    Classic reliability failure-time data from Aarset's bathtub-shaped hazard example.

    These positive failure times are often used in reliability-distribution papers.
    They provide a compact additional domain beyond biomedical and wine examples.
    """
    return np.array(
        [
            0.1,
            0.2,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            3.0,
            6.0,
            7.0,
            11.0,
            12.0,
            18.0,
            18.0,
            18.0,
            18.0,
            18.0,
            21.0,
            32.0,
            36.0,
            40.0,
            45.0,
            46.0,
            47.0,
            50.0,
            55.0,
            60.0,
            63.0,
            63.0,
            67.0,
            67.0,
            67.0,
            67.0,
            72.0,
            75.0,
            79.0,
            82.0,
            82.0,
            83.0,
            84.0,
            84.0,
            84.0,
            85.0,
            85.0,
            85.0,
            85.0,
            85.0,
            86.0,
            86.0,
        ],
        dtype=float,
    )


def plot_variance_mixing_laws(a=0.5, m=1.0, b=2.0):
    """Plot the NUVM uniform variance law and the TVMN triangular variance law."""
    v = np.linspace(a - 0.15 * (b - a), b + 0.15 * (b - a), 800)
    uniform_pdf = np.where((v >= a) & (v <= b), 1.0 / (b - a), 0.0)
    triangular_pdf = tvmn_core.triangular_variance_pdf(v, tvmn_core.TVMNParameters(a, m, b))

    plt.figure(figsize=(8.5, 5.0))
    plt.plot(v, uniform_pdf, linewidth=2.6, label=f"Uniform variance law on [{a}, {b}]")
    plt.plot(v, triangular_pdf, linewidth=2.6, label=f"Triangular variance law, mode {m}")
    plt.fill_between(v, 0.0, uniform_pdf, alpha=0.12)
    plt.fill_between(v, 0.0, triangular_pdf, alpha=0.12)
    plt.xlabel("Latent variance v")
    plt.ylabel("Mixing density g(v)")
    plt.title("Variance Mixing Laws: NUVM versus TVMN")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    path = FIGURE_DIR / "15_variance_mixing_laws_NUVM_TVMN.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def run_large_sample_recovery(sample_sizes=(1000, 2000, 5000), true_params=(0.5, 1.0, 2.0), seed=202606):
    """Estimate TVMN at larger n to demonstrate asymptotic recovery."""
    rows = []
    for n in sample_sizes:
        print(f"Large-sample recovery n={n}")
        data = tvmn_mle.generate_tvmn_sample(n, *true_params, seed=seed + n)
        fit = tvmn_mle.estimate_tvmn_mle(data, n_starts=32, seed=seed + 10_000 + n)
        loglik_true = tvmn_mle.loglik(true_params, data)
        rows.append(
            {
                "n": n,
                "true_a": true_params[0],
                "true_m": true_params[1],
                "true_b": true_params[2],
                "a_hat": fit["a_hat"],
                "m_hat": fit["m_hat"],
                "b_hat": fit["b_hat"],
                "bias_a": fit["a_hat"] - true_params[0],
                "bias_m": fit["m_hat"] - true_params[1],
                "bias_b": fit["b_hat"] - true_params[2],
                "m_minus_a": fit["m_hat"] - fit["a_hat"],
                "b_minus_m": fit["b_hat"] - fit["m_hat"],
                "loglik_hat": fit["loglik"],
                "loglik_true": loglik_true,
                "loglik_gain": fit["loglik"] - loglik_true,
                "success": fit["success"],
                "message": fit["message"],
            }
        )

    recovery = pd.DataFrame(rows)
    recovery.to_csv(OUTPUT_DIR / "06_TVMN_large_sample_recovery.csv", index=False)
    plot_large_sample_recovery(recovery, true_params)
    return recovery


def plot_large_sample_recovery(recovery, true_params):
    """Plot large-sample recovery estimates against true values."""
    names = ["a", "m", "b"]
    plt.figure(figsize=(8.8, 5.2))
    for idx, name in enumerate(names):
        plt.plot(recovery["n"], recovery[f"{name}_hat"], marker="o", linewidth=2.2, label=fr"$\hat{{{name}}}$")
        plt.axhline(true_params[idx], linestyle="--", linewidth=1.4, alpha=0.7, label=f"true {name}")
    plt.xlabel("Sample size")
    plt.ylabel("Parameter value")
    plt.title("TVMN Large-Sample MLE Recovery")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "16_TVMN_large_sample_recovery.png", dpi=300)
    plt.close()


def contour_grid_am(data, b_fixed, a_range, m_range, grid_size):
    a_values = np.linspace(a_range[0], a_range[1], grid_size)
    m_values = np.linspace(m_range[0], m_range[1], grid_size)
    z = np.full((grid_size, grid_size), np.nan)
    for i, m in enumerate(m_values):
        for j, a in enumerate(a_values):
            if 0.0 < a < m < b_fixed:
                z[i, j] = tvmn_mle.loglik((a, m, b_fixed), data)
    return a_values, m_values, z


def contour_grid_mb(data, a_fixed, m_range, b_range, grid_size):
    m_values = np.linspace(m_range[0], m_range[1], grid_size)
    b_values = np.linspace(b_range[0], b_range[1], grid_size)
    z = np.full((grid_size, grid_size), np.nan)
    for i, b in enumerate(b_values):
        for j, m in enumerate(m_values):
            if 0.0 < a_fixed < m < b:
                z[i, j] = tvmn_mle.loglik((a_fixed, m, b), data)
    return m_values, b_values, z


def plot_likelihood_contours(n=500, true_params=(0.5, 1.0, 2.0), seed=4242, grid_size=75):
    """
    Plot likelihood contours that show why finite-sample estimation is difficult.

    The contours use one simulated dataset. The (a, m) plot fixes b at the MLE,
    and the (m, b) plot fixes a at the MLE.
    """
    data = tvmn_mle.generate_tvmn_sample(n, *true_params, seed=seed)
    fit = tvmn_mle.estimate_tvmn_mle(data, n_starts=32, seed=seed + 1)
    mle = np.array([fit["a_hat"], fit["m_hat"], fit["b_hat"]], dtype=float)

    a_values, m_values, z_am = contour_grid_am(
        data=data,
        b_fixed=mle[2],
        a_range=(max(0.05, 0.25 * true_params[0]), min(1.4 * true_params[1], 0.95 * mle[2])),
        m_range=(max(0.10, 0.45 * true_params[1]), min(1.7 * true_params[2], 0.98 * mle[2])),
        grid_size=grid_size,
    )
    plot_contour(
        x_values=a_values,
        y_values=m_values,
        z=z_am,
        xlabel="a",
        ylabel="m",
        title=fr"TVMN Log-Likelihood Contour: $(a,m)$ with $b={mle[2]:.3f}$",
        filename="17_TVMN_likelihood_contour_am.png",
        true_point=(true_params[0], true_params[1]),
        mle_point=(mle[0], mle[1]),
    )

    m_values, b_values, z_mb = contour_grid_mb(
        data=data,
        a_fixed=mle[0],
        m_range=(max(0.10, 0.45 * true_params[1]), 1.6 * true_params[2]),
        b_range=(max(0.70, 0.75 * true_params[2]), 2.5 * true_params[2]),
        grid_size=grid_size,
    )
    plot_contour(
        x_values=m_values,
        y_values=b_values,
        z=z_mb,
        xlabel="m",
        ylabel="b",
        title=fr"TVMN Log-Likelihood Contour: $(m,b)$ with $a={mle[0]:.3f}$",
        filename="18_TVMN_likelihood_contour_mb.png",
        true_point=(true_params[1], true_params[2]),
        mle_point=(mle[1], mle[2]),
    )

    pd.DataFrame(
        [
            {
                "n": n,
                "true_a": true_params[0],
                "true_m": true_params[1],
                "true_b": true_params[2],
                "a_hat": mle[0],
                "m_hat": mle[1],
                "b_hat": mle[2],
                "loglik_hat": fit["loglik"],
            }
        ]
    ).to_csv(OUTPUT_DIR / "06_TVMN_likelihood_contour_fit.csv", index=False)

    return fit


def plot_contour(x_values, y_values, z, xlabel, ylabel, title, filename, true_point, mle_point):
    """Draw a relative log-likelihood contour plot."""
    z_relative = z - np.nanmax(z)
    levels = [-12.0, -8.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.1]

    plt.figure(figsize=(7.0, 5.6))
    contour = plt.contourf(x_values, y_values, z_relative, levels=levels, cmap="viridis", extend="min")
    lines = plt.contour(x_values, y_values, z_relative, levels=levels, colors="white", linewidths=0.75, alpha=0.75)
    plt.clabel(lines, inline=True, fontsize=8, fmt="%.1f")
    plt.scatter([true_point[0]], [true_point[1]], s=70, marker="x", color="white", linewidths=2.2, label="True")
    plt.scatter([mle_point[0]], [mle_point[1]], s=54, marker="o", color="#d62828", edgecolor="white", label="MLE")
    plt.colorbar(contour, label="Relative log-likelihood")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def run_reliability_benchmark(seed=6060):
    """Fit reviewer benchmark models to Aarset reliability failure times."""
    data = aarset_failure_times()
    name = "aarset_failure_times"
    fits = [
        empirical.fit_normal(data),
        empirical.fit_laplace(data),
        empirical.fit_student_t(data),
        empirical.fit_gamma(data),
        empirical.fit_lognormal(data),
        empirical.fit_weibull(data),
        empirical.fit_exponential(data),
        empirical.fit_nuvm(data, seed=seed),
        empirical.fit_tvmn_location(data, seed=seed + 100),
    ]

    rows = []
    parameter_rows = []
    for fit in fits:
        _, cdf_func, _ = empirical.model_pdf_cdf_sf(fit)
        rows.append(
            {
                "dataset": name,
                "n": len(data),
                "model": fit["model"],
                "loglik": fit["loglik"],
                "k": fit["k"],
                "KS": empirical.empirical_ks(data, cdf_func),
                **empirical.information_criteria(fit["loglik"], fit["k"], len(data)),
            }
        )
        parameter_rows.append({"dataset": name, "model": fit["model"], **fit["params"]})

    comparison = pd.DataFrame(rows).sort_values("AIC")
    parameters = pd.DataFrame(parameter_rows)
    comparison.to_csv(OUTPUT_DIR / "06_reliability_aarset_model_comparison.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "06_reliability_aarset_parameter_estimates.csv", index=False)

    empirical.plot_benchmark_fit(name, data, fits)
    return comparison, parameters


def write_notes(args):
    notes = [
        "# TVMN Reviewer-Strengthening Outputs",
        "",
        "Generated by `06_TVMN_reviewer_strengthening.py`.",
        "",
        "## Files to add to the manuscript",
        "",
        "- `15_variance_mixing_laws_NUVM_TVMN.png`: visual comparison of NUVM and TVMN mixing laws.",
        "- `16_TVMN_large_sample_recovery.png` and `06_TVMN_large_sample_recovery.csv`: recovery at larger sample sizes.",
        "- `17_TVMN_likelihood_contour_am.png`: likelihood surface for `(a,m)` with `b` fixed at the MLE.",
        "- `18_TVMN_likelihood_contour_mb.png`: likelihood surface for `(m,b)` with `a` fixed at the MLE.",
        "- `06_reliability_aarset_model_comparison.csv`: additional reliability/failure-time application.",
        "",
        "## Suggested manuscript wording",
        "",
        "The large-sample recovery and likelihood-contour figures should be presented as evidence that the",
        "TVMN likelihood is valid but can be flat in finite samples. The reliability benchmark should be framed",
        "as an additional applied domain, not as proof of universal superiority.",
        "",
        "## Current run settings",
        "",
        f"- part: {args.part}",
        f"- recovery sizes: {args.recovery_sizes}",
        f"- contour n: {args.contour_n}",
        f"- contour grid size: {args.grid_size}",
        f"- seed: {args.seed}",
    ]
    path = OUTPUT_DIR / "06_TVMN_reviewer_strengthening_notes.md"
    path.write_text("\n".join(notes), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Reviewer-strengthening TVMN additions.")
    parser.add_argument(
        "--part",
        choices=["all", "mixing", "recovery", "contours", "reliability"],
        default="all",
    )
    parser.add_argument("--recovery-sizes", type=int, nargs="+", default=[1000, 2000, 5000])
    parser.add_argument("--contour-n", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=202606)
    args = parser.parse_args()

    ensure_output_dirs()

    if args.part in ["all", "mixing"]:
        print("Plotting variance mixing laws...")
        print(plot_variance_mixing_laws())

    if args.part in ["all", "recovery"]:
        print("Running large-sample recovery...")
        print(run_large_sample_recovery(sample_sizes=args.recovery_sizes, seed=args.seed))

    if args.part in ["all", "contours"]:
        print("Plotting likelihood contours...")
        print(plot_likelihood_contours(n=args.contour_n, seed=args.seed + 1, grid_size=args.grid_size))

    if args.part in ["all", "reliability"]:
        print("Running reliability benchmark...")
        comparison, _ = run_reliability_benchmark(seed=args.seed + 2)
        print(comparison[["model", "loglik", "AIC", "BIC", "KS"]])

    notes_path = write_notes(args)
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()
