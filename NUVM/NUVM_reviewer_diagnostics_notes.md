# NUVM Reviewer Diagnostics

## Why a_hat = 0.999 Matters

For the NUVM model,

```text
V ~ Uniform((1-a)sigma^2, (1+a)sigma^2).
```

When `a` is close to 1, the variance interval is almost

```text
(0, 2 sigma^2).
```

Therefore, `a_hat = 0.999` means the fitted model is using nearly the maximum allowed variance uncertainty. This can happen because the data are genuinely heavy-tailed, because the likelihood is nearly flat near the upper boundary, or because NUVM is too light-tailed for the dataset.

## Likelihood Profile Diagnostic

The profile likelihood is

```text
l_p(a) = max_{mu, sigma} l(mu, sigma, a).
```

If `l_p(a)` is flat near 1, the issue is likely weak identifiability between `sigma` and `a`.

If `l_p(a)` sharply increases toward 1, the boundary estimate is a genuine maximum for this dataset.

Run:

```powershell
python NUVM_Reviewer_Diagnostics.py --part profile
```

Output:

```text
NUVM_likelihood_profile_NIFTY.csv
figures/NUVM_likelihood_profile_NIFTY.png
figures/NUVM_histogram_densities_NIFTY.png
figures/NUVM_QQ_plots_NIFTY.png
```

## Method of Moments Estimator

For NUVM,

```text
E[X] = mu
Var(X) = sigma^2
Kurtosis(X) = 3 + a^2.
```

Thus,

```text
mu_hat = sample mean
sigma_hat = sample standard deviation
a_hat = sqrt(max(sample kurtosis - 3, 0)).
```

Since NUVM requires `0 < a < 1`, the estimator is clipped to `[0.0001, 0.999]`.

If the raw MoM estimate is above 1, the sample kurtosis is outside the NUVM fourth-moment range. This is useful evidence that the data are heavier-tailed than NUVM can naturally represent.

Run:

```powershell
python NUVM_Reviewer_Diagnostics.py --part mom
```

## Additional Datasets

Run:

```powershell
python NUVM_Reviewer_Diagnostics.py --part multi
```

This fits Normal, Student-t, NUVM-MLE, and NUVM-MoM to:

```text
NIFTY
S&P 500
NASDAQ
Gold
Crude Oil
```

Output:

```text
NUVM_multi_dataset_comparison.csv
```

## Simulated Recovery of a

Run:

```powershell
python NUVM_Reviewer_Diagnostics.py --part recovery --replications 1000
```

Output:

```text
NUVM_a_recovery_raw.csv
NUVM_a_recovery_summary.csv
```

This creates the table:

```text
True a | Mean MLE a_hat | RMSE MLE a_hat | Mean MoM a_hat | RMSE MoM a_hat | Convergence Rate
```

## All Reviewer Diagnostics

Run:

```powershell
python NUVM_Reviewer_Diagnostics.py --part all --replications 1000
```

This runs the profile, recovery study, multi-dataset study, and RMSE figure generation.
