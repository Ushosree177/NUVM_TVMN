# TVMN/NUVM Reviewer-Readiness Audit Checklist

## Theory

- [x] Definition of distribution.
- [x] PDF.
- [x] CDF.
- [x] Survival function.
- [x] Hazard function.
- [x] Reverse hazard function.
- [x] Cumulative hazard function.
- [ ] Closed-form quantile function.
- [x] Numerical quantile computation can be added through CDF inversion.
- [x] Moments.
- [x] Kurtosis.
- [x] Numerical verification that the PDF integrates to one.

## Estimation

- [x] MLE likelihood.
- [x] Log-likelihood.
- [x] Numerical optimization procedure.
- [x] Parameter constraints enforced by transformation.
- [x] Fisher information and Wald intervals.
- [x] Bootstrap confidence intervals.

## Simulation

- [x] Multiple sample sizes.
- [x] Bias.
- [x] MSE.
- [x] RMSE.
- [x] Convergence summaries.

## Real Data

- [x] Multiple datasets.
- [x] Competing distributions.
- [x] Log-likelihood.
- [x] AIC/BIC/CAIC/HQIC.
- [x] KS diagnostics.
- [x] AD/CVM diagnostics computed in `TVMN_NUVM_submission_ready_results.ipynb`.
- [x] Histogram with fitted density.
- [x] Empirical CDF comparison.
- [x] Hazard shape plots.

## Current Reviewer Position

The strongest position is that TVMN is mathematically valid, estimable, and competitive with NUVM in selected datasets. The current evidence does not support a universal superiority claim over all classical alternatives.
