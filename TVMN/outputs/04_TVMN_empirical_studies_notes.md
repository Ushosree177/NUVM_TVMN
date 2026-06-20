# TVMN Empirical Studies Notes

This script covers Fisher information, Monte Carlo study, real-data applications, and bootstrap.

## Recommended Paper-Scale Runs

```powershell
python TVMN\04_TVMN_empirical_studies.py --part monte-carlo --mc-replications 1000
python TVMN\04_TVMN_empirical_studies.py --part bootstrap --bootstrap-replications 1000
python TVMN\04_TVMN_empirical_studies.py --part real-data
```

## Current Run Settings

- part: benchmark
- Monte Carlo replications: 50
- bootstrap replications: 200
- bootstrap n: 250
- Fisher n: 500

## Interpretation Outputs

- `04_TVMN_real_data_interpretation.md`: reviewer-facing summary of the mixed real-data evidence.
- `04_TVMN_real_data_winners.csv`: information-criterion winners and TVMN deltas.
- `04_TVMN_vs_NUVM_likelihood_comparison.csv`: descriptive TVMN-versus-NUVM likelihood differences.
- `05_TVMN_benchmark_model_comparison.csv`: expanded benchmark comparison against 8 competitors.
- `05_TVMN_benchmark_interpretation.md`: reviewer-facing benchmark interpretation.

## Publication Framing

Use a competitive-performance claim for the current manuscript. The real-data evidence does not yet support a universal superiority claim for TVMN.

The existing Monte Carlo and bootstrap CSV files may come from larger paper-scale runs than the settings shown above if only one section was regenerated.
