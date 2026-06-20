# NUVM Paper Additions

## Sixth Central Moment

Let

```text
X | V ~ N(mu, V),
V ~ Uniform((1-a)sigma^2, (1+a)sigma^2).
```

For a normal random variable,

```text
E[(X - mu)^6 | V] = 15 V^3.
```

Therefore,

```text
E[(X - mu)^6] = 15 E[V^3].
```

Since

```text
E[V^3]
= {U^4 - L^4} / {4(U - L)}
= sigma^6(1 + a^2),
```

where `L = (1-a)sigma^2` and `U = (1+a)sigma^2`, the sixth central moment is

```text
E[(X - mu)^6] = 15 sigma^6(1 + a^2).
```

## Eighth Central Moment

For a normal random variable,

```text
E[(X - mu)^8 | V] = 105 V^4.
```

Therefore,

```text
E[(X - mu)^8] = 105 E[V^4].
```

Since

```text
E[V^4]
= {U^5 - L^5} / {5(U - L)}
= sigma^8(1 + 2a^2 + a^4/5),
```

the eighth central moment is

```text
E[(X - mu)^8] = 105 sigma^8(1 + 2a^2 + a^4/5).
```

## Corrected MLE

The corrected MLE uses direct parameters and bounded L-BFGS-B:

```python
bounds = [
    (min(data) - 5 * std(data), max(data) + 5 * std(data)),
    (1e-6, 10 * std(data)),
    (1e-4, 0.999),
]

result = minimize(
    negative_loglik_bounded,
    x0=start,
    args=(data,),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
)
```

This replaces the unstable transformed BFGS code:

```python
sigma = exp(theta[1])
a = 1 / (1 + exp(-theta[2]))
```

## Commands

Run the full simulation study:

```powershell
python 05_NUVM_Simulation_and_Real_Data.py --part simulation --replications 1000
```

Run the real-data study with 1000 bootstrap samples:

```powershell
python 05_NUVM_Simulation_and_Real_Data.py --part real --bootstrap-replications 1000
```

Run a faster real-data check without bootstrap:

```powershell
python 05_NUVM_Simulation_and_Real_Data.py --part real --bootstrap-replications 0
```

## New Output Tables

```text
05_NUVM_higher_moments.csv
05_NUVM_observed_fisher_information.csv
05_NUVM_wald_confidence_intervals.csv
05_NUVM_bootstrap_raw_results.csv
05_NUVM_bootstrap_confidence_intervals.csv
05_NUVM_goodness_of_fit.csv
```
