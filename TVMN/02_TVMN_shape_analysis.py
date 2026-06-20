"""
TVMN Shape Analysis

This script continues after the completed foundation stage.

Today only:
    1. PDF shape plots for several (a, m, b) combinations.
    2. Tail-view plots on x in [-5, 5] and [-10, 10].
    3. TVMN vs Normal comparison with matched variance.
    4. TVMN vs NUVM comparison.
    5. Simulated TVMN histogram with theoretical PDF overlay.

No MLE, Fisher information, simulation study, real data, or bootstrap is done here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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


tvmn_core = load_module("tvmn_core", BASE_DIR / "01_TVMN_step_by_step_derivation.py")
nuvm_core = load_module("nuvm_core", PROJECT_DIR / "NUVM" / "NUVM_MLE.py")


def ensure_output_dirs():
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def tvmn_variance(a, m, b):
    """Var(X) for TVMN because E[X]=0 and E[X^2]=E[V]."""
    return (a + m + b) / 3.0


def generate_tvmn_sample(n, a, m, b, seed=123):
    """
    Generate X from TVMN.

    Algorithm:
        1. Generate V ~ Triangular(a, m, b).
        2. Generate X | V ~ Normal(0, V).
        3. Return X.
    """
    rng = np.random.default_rng(seed)
    v = rng.triangular(left=a, mode=m, right=b, size=n)
    x = rng.normal(loc=0.0, scale=np.sqrt(v), size=n)
    return x


def plot_pdf_shapes(parameter_sets, x_min, x_max, filename, title):
    """Plot TVMN PDFs for many parameter combinations."""
    x = np.linspace(x_min, x_max, 1200)

    plt.figure(figsize=(9, 5.5))
    for a, m, b in parameter_sets:
        pdf = tvmn_core.tvmn_pdf(x, a, m, b)
        plt.plot(x, pdf, linewidth=2.0, label=f"a={a}, m={m}, b={b}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_tvmn_vs_normal(a, m, b):
    """
    Compare TVMN with a Normal distribution having the same variance.

    Normal comparator:
        N(0, Var_TVMN)
    """
    x = np.linspace(-6, 6, 1200)
    variance = tvmn_variance(a, m, b)
    sigma = np.sqrt(variance)

    tvmn_pdf = tvmn_core.tvmn_pdf(x, a, m, b)
    normal_pdf = stats.norm.pdf(x, loc=0.0, scale=sigma)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(x, tvmn_pdf, linewidth=2.4, label=f"TVMN({a}, {m}, {b})")
    plt.plot(x, normal_pdf, linestyle="--", linewidth=2.4, label=f"Normal(0, {variance:.4f})")
    plt.title("TVMN vs Normal PDF: Matched Variance")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "03_TVMN_vs_Normal_matched_variance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_tvmn_vs_nuvm(a, m, b, nuvm_alpha=0.50):
    """
    Compare TVMN with NUVM on the same graph.

    NUVM parameterization used in the existing project:
        V ~ Uniform((1-alpha)sigma^2, (1+alpha)sigma^2)
        X | V ~ Normal(mu, V)

    Here sigma^2 is set equal to Var(TVMN), so both models have mean 0
    and the same unconditional variance. The NUVM alpha controls the
    uniform variance spread.
    """
    x = np.linspace(-6, 6, 1200)
    variance = tvmn_variance(a, m, b)
    sigma = np.sqrt(variance)

    tvmn_pdf = tvmn_core.tvmn_pdf(x, a, m, b)
    nuvm_pdf = nuvm_core.nuvm_pdf(x, mu=0.0, sigma=sigma, a=nuvm_alpha)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(x, tvmn_pdf, linewidth=2.4, label=f"TVMN({a}, {m}, {b})")
    plt.plot(
        x,
        nuvm_pdf,
        linestyle="--",
        linewidth=2.4,
        label=f"NUVM(mu=0, sigma={sigma:.4f}, alpha={nuvm_alpha})",
    )
    plt.title("TVMN vs NUVM PDF: Same Mean and Variance")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "04_TVMN_vs_NUVM_matched_variance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_simulated_histogram(a, m, b, n=10000, seed=123):
    """Plot simulated TVMN histogram with theoretical PDF overlay."""
    sample = generate_tvmn_sample(n, a, m, b, seed=seed)
    x_min, x_max = np.percentile(sample, [0.2, 99.8])
    pad = 0.8
    x = np.linspace(x_min - pad, x_max + pad, 1200)
    theoretical_pdf = tvmn_core.tvmn_pdf(x, a, m, b)

    plt.figure(figsize=(8.5, 5.2))
    plt.hist(sample, bins=60, density=True, alpha=0.42, color="#6b8dbd", edgecolor="white", label="Simulated data")
    plt.plot(x, theoretical_pdf, color="#b23a48", linewidth=2.6, label="Theoretical TVMN PDF")
    plt.title(f"TVMN Simulated Histogram with PDF Overlay, n={n}")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "05_TVMN_histogram_theoretical_pdf.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path, sample


def write_observations(parameter_sets, base_params, sample):
    """Write reviewer-friendly observations from the shape analysis."""
    a, m, b = base_params
    variance = tvmn_variance(a, m, b)
    sigma = np.sqrt(variance)

    rows = []
    for p in parameter_sets:
        pa, pm, pb = p
        x0_pdf = tvmn_core.tvmn_pdf(0.0, pa, pm, pb)
        tail5_pdf = tvmn_core.tvmn_pdf(5.0, pa, pm, pb)
        tail10_pdf = tvmn_core.tvmn_pdf(10.0, pa, pm, pb)
        rows.append(
            {
                "a": pa,
                "m": pm,
                "b": pb,
                "Var(X)": tvmn_variance(pa, pm, pb),
                "f(0)": x0_pdf,
                "f(5)": tail5_pdf,
                "f(10)": tail10_pdf,
            }
        )

    csv_path = OUTPUT_DIR / "02_TVMN_shape_summary.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("a,m,b,Var(X),f(0),f(5),f(10)\n")
        for row in rows:
            handle.write(
                f"{row['a']},{row['m']},{row['b']},{row['Var(X)']},"
                f"{row['f(0)']},{row['f(5)']},{row['f(10)']}\n"
            )

    markdown_path = OUTPUT_DIR / "02_TVMN_shape_observations.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# TVMN Shape Analysis Observations\n\n")
        handle.write("This file records observations only for the requested shape-analysis stage.\n\n")
        handle.write("## PDF Shape\n\n")
        handle.write("- TVMN densities are symmetric around zero because only the variance is mixed and E[X | V]=0.\n")
        handle.write("- Smaller variance intervals concentrate probability near zero and produce a sharper peak.\n")
        handle.write("- Larger maximum variance b increases tail thickness because more mass is assigned to larger conditional variances.\n")
        handle.write("- Moving the mode m toward larger values shifts more mixing weight toward larger variances, flattening the center and lifting the tails.\n\n")
        handle.write("## Tail Study\n\n")
        handle.write("- The [-5, 5] view shows central peakedness and moderate-tail behavior.\n")
        handle.write("- The [-10, 10] view makes the far-tail differences clearer; parameter sets with larger b retain visibly higher tail density.\n\n")
        handle.write("## TVMN vs Normal\n\n")
        handle.write(f"- Base TVMN parameters: a={a}, m={m}, b={b}.\n")
        handle.write(f"- Matched Normal comparator: N(0, {variance:.6f}), with standard deviation {sigma:.6f}.\n")
        handle.write("- The TVMN curve can be more peaked near zero and heavier in the tails than the variance-matched Normal because it is a variance mixture.\n\n")
        handle.write("## TVMN vs NUVM\n\n")
        handle.write("- NUVM is the uniform variance mixture predecessor; TVMN replaces the uniform variance law with a triangular variance law.\n")
        handle.write("- In the comparison plot, TVMN and NUVM are matched by unconditional mean and variance.\n")
        handle.write("- TVMN gives extra control through m, the most likely variance, which NUVM does not have.\n\n")
        handle.write("## Simulation Histogram\n\n")
        handle.write("- The simulated sample is generated by first drawing V from the triangular variance distribution and then drawing X from N(0,V).\n")
        handle.write("- The histogram aligns with the theoretical PDF, supporting the random-generation algorithm.\n\n")
        handle.write("## Simulated Sample Summary\n\n")
        handle.write(f"- n = {len(sample)}\n")
        handle.write(f"- sample mean = {np.mean(sample):.6f}\n")
        handle.write(f"- sample variance = {np.var(sample, ddof=1):.6f}\n")
        handle.write(f"- theoretical variance = {variance:.6f}\n")

    return csv_path, markdown_path


def main():
    ensure_output_dirs()

    parameter_sets = [
        (0.5, 1.0, 2.0),
        (0.2, 1.0, 1.8),
        (0.8, 1.0, 1.2),
        (0.5, 1.5, 3.0),
        (1.0, 2.0, 5.0),
        (0.3, 0.8, 2.5),
        (0.7, 1.8, 2.2),
    ]
    base_params = (0.5, 1.0, 2.0)

    print("TVMN Stage 2 Shape Analysis")
    print("=" * 70)
    print("No foundation derivation, MLE, simulation study, real data, or bootstrap is run here.")

    saved_files = []
    saved_files.append(
        plot_pdf_shapes(
            parameter_sets,
            -5,
            5,
            "01_TVMN_pdf_shapes_x_minus5_to_5.png",
            "TVMN PDF Shapes on x in [-5, 5]",
        )
    )
    saved_files.append(
        plot_pdf_shapes(
            parameter_sets,
            -10,
            10,
            "02_TVMN_pdf_tail_view_x_minus10_to_10.png",
            "TVMN PDF Tail View on x in [-10, 10]",
        )
    )
    saved_files.append(plot_tvmn_vs_normal(*base_params))
    saved_files.append(plot_tvmn_vs_nuvm(*base_params, nuvm_alpha=0.50))
    histogram_path, sample = plot_simulated_histogram(*base_params, n=10000, seed=123)
    saved_files.append(histogram_path)

    summary_csv, observations_md = write_observations(parameter_sets, base_params, sample)
    saved_files.extend([summary_csv, observations_md])

    print("\nSaved outputs:")
    for path in saved_files:
        print(path)

    print("\nCore observations:")
    print("- TVMN is symmetric around zero.")
    print("- Larger b or larger m generally gives flatter centers and heavier tails.")
    print("- Smaller variance ranges give sharper peaks and lighter tails.")
    print("- Compared with Normal, TVMN can be more peaked and heavier-tailed.")
    print("- Compared with NUVM, TVMN adds the mode m, giving more flexible variance mixing.")


if __name__ == "__main__":
    main()
