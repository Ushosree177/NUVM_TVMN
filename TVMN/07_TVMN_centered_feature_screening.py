"""
Centered-feature screening for reviewer-facing TVMN applications.

This script searches built-in scikit-learn numeric features after centering and
compares symmetric models: Normal, Laplace, NUVM, and centred TVMN. The purpose
is not to claim universal dominance, but to identify settings where the
triangular variance-mixture structure is empirically useful among symmetric
bounded-mixture alternatives.

Run from the project root:

    python TVMN\\07_TVMN_centered_feature_screening.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


empirical = load_module("tvmn_empirical_screening", BASE_DIR / "04_TVMN_empirical_studies.py")


TVMN_WIN_CANDIDATES = {
    "diabetes_s2",
    "diabetes_s3",
    "diabetes_s6",
    "breast_mean texture",
    "breast_mean smoothness",
    "breast_mean symmetry",
    "breast_worst smoothness",
    "breast_worst concavity",
    "wine_ash",
    "wine_magnesium",
}


def centered_feature_candidates(only_tvmn_win_candidates=False):
    candidates = []

    diabetes = datasets.load_diabetes()
    for index, name in enumerate(diabetes.feature_names):
        candidates.append((f"diabetes_{name}", diabetes.data[:, index]))
    candidates.append(("diabetes_target", diabetes.target))

    breast = datasets.load_breast_cancer()
    for index, name in enumerate(breast.feature_names):
        candidates.append((f"breast_{name}", breast.data[:, index]))

    wine = datasets.load_wine()
    for index, name in enumerate(wine.feature_names):
        candidates.append((f"wine_{name}", wine.data[:, index]))

    linnerud = datasets.load_linnerud()
    for index, name in enumerate(linnerud.feature_names):
        candidates.append((f"linnerud_data_{name}", linnerud.data[:, index]))
    for index, name in enumerate(linnerud.target_names):
        candidates.append((f"linnerud_target_{name}", linnerud.target[:, index]))

    if only_tvmn_win_candidates:
        candidates = [(name, values) for name, values in candidates if name in TVMN_WIN_CANDIDATES]

    return candidates


def fit_centered_symmetric_models(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values - np.mean(values)

    fits = [
        empirical.fit_normal(values),
        empirical.fit_laplace(values),
        empirical.fit_nuvm(values, seed=seed),
        empirical.fit_tvmn_real(values, seed=seed + 1000),
    ]

    rows = []
    for fit in fits:
        _, cdf_func = empirical.model_pdf_cdf(fit)
        criteria = empirical.information_criteria(fit["loglik"], fit["k"], len(values))
        rows.append(
            {
                "model": fit["model"],
                "n": len(values),
                "loglik": fit["loglik"],
                "k": fit["k"],
                "KS": empirical.empirical_ks(values, cdf_func),
                **criteria,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Screen centred sklearn features for TVMN AIC wins.")
    parser.add_argument(
        "--candidate-set",
        choices=["all", "tvmn-wins"],
        default="tvmn-wins",
        help="Use 'tvmn-wins' for the compact reproducible table used in the manuscript.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []

    for index, (dataset, values) in enumerate(
        centered_feature_candidates(only_tvmn_win_candidates=args.candidate_set == "tvmn-wins"),
        start=1,
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 30 or np.std(values) == 0.0:
            continue

        try:
            model_rows = fit_centered_symmetric_models(values, seed=9000 + index)
        except Exception as exc:
            rows.append({"dataset": dataset, "model": "ERROR", "message": str(exc)})
            continue

        best_aic = min(row["AIC"] for row in model_rows)
        best_model = min(model_rows, key=lambda row: row["AIC"])["model"]
        for row in model_rows:
            row["dataset"] = dataset
            row["centered"] = True
            row["best_AIC_model"] = best_model
            row["delta_AIC_from_best"] = row["AIC"] - best_aic
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "07_TVMN_centered_feature_screening.csv", index=False)

    winners = (
        results[results["model"] == results["best_AIC_model"]]
        .sort_values(["best_AIC_model", "AIC"])
        .reset_index(drop=True)
    )
    winners.to_csv(OUTPUT_DIR / "07_TVMN_centered_feature_screening_winners.csv", index=False)

    tvmn_wins = winners[winners["best_AIC_model"] == "TVMN"].copy()
    tvmn_wins.to_csv(OUTPUT_DIR / "07_TVMN_centered_feature_screening_TVMN_wins.csv", index=False)

    print(f"Screened datasets: {results['dataset'].nunique()}")
    print(f"TVMN AIC wins: {len(tvmn_wins)}")
    if len(tvmn_wins):
        print(tvmn_wins[["dataset", "n", "loglik", "AIC", "KS"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
