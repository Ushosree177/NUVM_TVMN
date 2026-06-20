# TVMN Good Result Summary

The expanded benchmark does not support a claim that TVMN dominates all competing distributions by AIC. However, it does provide useful positive results that can be reported honestly.

## Best Defensible Positive Result

TVMN improves over NUVM in several direct comparisons:

- Higher log-likelihood than NUVM in 5 out of 8 benchmark datasets.
- Smaller KS statistic than NUVM in 6 out of 8 benchmark datasets.
- Better AIC and BIC than NUVM for the breast cancer mean area dataset.

The strongest TVMN-versus-NUVM case is:

| Dataset | TVMN logLik - NUVM logLik | TVMN AIC - NUVM AIC | TVMN BIC - NUVM BIC |
|---|---:|---:|---:|
| breast_cancer_mean_area | 8.139 | -14.278 | -9.934 |

## Careful Manuscript Wording

> In the expanded benchmark study, TVMN did not uniformly dominate all classical competitors by AIC. However, relative to NUVM, TVMN produced higher likelihoods in five of eight datasets and smaller KS statistics in six of eight datasets. For the breast cancer mean area data, TVMN improved both AIC and BIC relative to NUVM, indicating that the triangular variance-mixture structure can provide practical gains over the uniform variance-mixture model in some settings.
