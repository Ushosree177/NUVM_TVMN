# TVMN Paper Additions and Improved Framing

## Suggested Empirical Claim

The empirical results support a competitive-performance claim, not a dominance claim.

Recommended wording:

> The TVMN distribution provides a mathematically tractable three-parameter variance-mixture model. Its maximum likelihood estimators show decreasing RMSE as the sample size increases in Monte Carlo experiments, and the fitted model gives competitive performance on several real datasets when compared with Normal, Laplace, and NUVM alternatives.

Avoid wording such as:

> TVMN is superior to existing models.

The present real-data evidence does not support a universal superiority claim.

## Theory Section

The theoretical section is strong and can be presented as a main contribution. The following elements are suitable for the main manuscript:

- Definition of the TVMN density.
- Closed-form or numerically verified PDF, CDF, survival, hazard, reverse hazard, and cumulative hazard functions.
- Moment derivations, including kurtosis.
- Verification that the PDF integrates to one.

Suggested manuscript wording:

> These results establish TVMN as a valid probability model with tractable distributional functions and computable moment structure. The availability of the CDF, survival, and hazard-related functions also makes the distribution suitable for reliability and lifetime-data applications.

## Monte Carlo Section

The Monte Carlo evidence supports consistency-like behavior of the MLE. Across sample sizes n = 30, 50, 100, 200, and 500, the RMSE decreases for all three parameters.

Suggested manuscript wording:

> The Monte Carlo study shows that the RMSE of the MLE decreases as the sample size increases. This pattern suggests that the likelihood-based estimation procedure becomes more stable at moderate sample sizes and is consistent with desirable large-sample behavior.

The current paper-scale output uses 1000 replications and should be preferred over earlier quick verification runs.

## Fisher Information and Uncertainty

The observed Fisher information results show that the upper variance parameter is better identified than the lower and modal variance parameters in the examined setting.

Suggested manuscript wording:

> The standard errors indicate relatively large uncertainty for the lower and modal variance parameters, while the upper variance parameter is more sharply identified. This pattern is consistent with the bootstrap results and suggests that moderate sample sizes may be needed for stable estimation of all three TVMN parameters.

This is not fatal, but it should be reported transparently.

## MLE Recovery Caveat

Earlier recovery checks showed that for n = 100 and n = 200 the likelihood may prefer a narrow triangular variance interval. For n = 500, recovery improves substantially.

Suggested manuscript wording:

> In small and moderate samples, the likelihood surface can favor near-degenerate triangular variance intervals. This behavior reflects finite-sample identifiability challenges rather than invalidity of the model. The Monte Carlo results indicate improved recovery as n increases, so empirical applications should preferably use moderate or large samples.

## Real-Data Results

The current real-data comparison is mixed:

| Dataset | Best model by AIC | TVMN result |
|---|---|---|
| Diabetes target | Normal | TVMN loses narrowly to Normal and NUVM |
| Breast cancer mean radius | NUVM | TVMN is competitive but not best |
| Breast cancer mean area | Laplace | TVMN is not competitive here |
| Wine alcohol | Normal | TVMN is nearly tied but not best |
| Linnerud weight | Laplace | n = 20, diagnostic only |

Overall, TVMN does not currently win AIC on the substantive real-data examples. This should be framed honestly.

Suggested manuscript wording:

> The real-data applications indicate that TVMN is competitive in some cases, especially where it is close to the best information criterion value, but it does not uniformly dominate the benchmark distributions. These findings suggest that TVMN should be viewed as an additional flexible model in the analyst's toolbox rather than as a replacement for all standard alternatives.

## TVMN Versus NUVM

TVMN and NUVM should be compared by log-likelihood and information criteria. A formal likelihood-ratio test should not be claimed unless a nested model relationship is established.

Suggested manuscript wording:

> Because the implemented TVMN and NUVM specifications are non-nested, likelihood differences are reported descriptively rather than as formal likelihood-ratio tests. Information criteria are used as the primary model-comparison tools.

## Publication Strategy

The strongest current paper position is:

> TVMN is a new mathematically valid and tractable variance-mixture distribution with a working likelihood-based estimation framework and competitive empirical behavior.

Current strength:

| Component | Assessment |
|---|---|
| Mathematical novelty | Strong |
| Simulation evidence | Strong |
| Estimation framework | Good |
| Real-data superiority | Weak |
| Overall positioning | Q3 to Q2 borderline unless empirical evidence is expanded |

## Recommended Improvements Before Submission

Add 10-20 additional datasets, especially from domains where variance mixtures are naturally useful:

- Finance returns.
- Insurance losses.
- Rainfall or hydrology data.
- Reliability and lifetime data.
- Environmental extremes.

For each dataset, report:

- Parameter estimates.
- Log-likelihood.
- AIC, BIC, CAIC, and HQIC.
- KS statistic or another goodness-of-fit diagnostic.
- Histogram with fitted PDFs.
- Empirical CDF comparison.
- TVMN Q-Q plot.

The empirical section becomes much stronger if TVMN wins at least some datasets by AIC or BIC and shows visible tail-fit advantages.
