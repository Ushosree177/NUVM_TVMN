# TVMN Strong Benchmark Evidence

This section compares TVMN and NUVM against Normal, Laplace, Student-t, Gamma, Lognormal, Weibull, and Exponential models on raw positive datasets.

## AIC Winners

| Dataset | n | AIC winner | TVMN AIC - best AIC |
|---|---:|---|---:|
| diabetes_target | 442 | Gamma | 60.240 |
| breast_cancer_mean_radius | 569 | Lognormal | 75.996 |
| breast_cancer_mean_texture | 569 | Lognormal | 37.914 |
| breast_cancer_mean_area | 569 | Lognormal | 189.468 |
| wine_alcohol | 178 | Normal | 4.001 |
| wine_malic_acid | 178 | Lognormal | 61.396 |
| wine_proline | 178 | Lognormal | 40.028 |
| linnerud_weight | 20 | Laplace | 3.859 |

TVMN AIC wins: 0/8.

## TVMN Versus NUVM

TVMN has higher log-likelihood than NUVM in 5/8 benchmark datasets.

TVMN has a smaller KS statistic than NUVM in 6/8 benchmark datasets.

The detailed comparison is saved in `05_TVMN_benchmark_TVMN_vs_NUVM.csv`.

## Interpretation

These expanded comparisons are the main reviewer-facing evidence. If TVMN wins or nearly ties on some datasets, the manuscript can claim competitive empirical performance. If it does not win, the paper should emphasize mathematical novelty, valid estimation, and situations where variance-mixture flexibility is practically useful.

## Recommended Wording

> Across several benchmark datasets, TVMN was compared with common symmetric, heavy-tailed, and positive-support distributions using log-likelihood and information criteria. The results show where the additional triangular variance-mixture structure is empirically useful and where simpler alternatives remain preferable.
