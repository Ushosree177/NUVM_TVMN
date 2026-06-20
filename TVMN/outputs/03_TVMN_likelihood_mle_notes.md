# TVMN Likelihood and MLE Notes

This stage defines and tests likelihood-based estimation only.

## Likelihood

For independent observations `x_1,...,x_n`,

```text
L(a,m,b) = product_{i=1}^n f(x_i; a,m,b)
```

where `f(x_i; a,m,b)` is the TVMN PDF derived in Stage 1.

## Log-likelihood

```text
ell(a,m,b) = sum_{i=1}^n log f(x_i; a,m,b)
```

The numerical optimization minimizes the negative log-likelihood.

## Constraints

```text
a > 0
a < m < b
```

Internally, the optimizer uses the positive-gap transformation:

```text
a = exp(theta_0)
m = a + exp(theta_1)
b = m + exp(theta_2)
```

This automatically preserves `a > 0` and `a < m < b`.

## Artificial Data Recovery

True parameters: `a=0.5`, `m=1.0`, `b=2.0`.

| n | a_hat | m_hat | b_hat | loglik |
|---:|---:|---:|---:|---:|
| 100 | 1.144513 | 1.163456 | 1.172107 | -149.347079 |
| 200 | 1.065958 | 1.105019 | 1.110541 | -292.740158 |
| 500 | 0.804522 | 0.839295 | 2.002536 | -758.008746 |

## Recovery Diagnostic

The table below compares the log-likelihood at the MLE with the log-likelihood at the true data-generating parameters.

| n | loglik_hat | loglik_true | gain | m_hat-a_hat | b_hat-m_hat |
|---:|---:|---:|---:|---:|---:|
| 100 | -149.347079 | -149.467859 | 0.120780 | 0.018943 | 0.008651 |
| 200 | -292.740158 | -293.668988 | 0.928830 | 0.039061 | 0.005523 |
| 500 | -758.008746 | -758.267240 | 0.258495 | 0.034773 | 1.163241 |

For smaller samples, the likelihood may prefer a very narrow triangular variance interval. This is a useful warning for the next stage: the finite-sample behavior of the three TVMN variance parameters must be studied carefully by Monte Carlo simulation.

These runs are a first recovery check. The formal Monte Carlo bias, MSE, and RMSE study belongs to the next stage.
