# NUVM and TVMN Distributions

This repository contains the code, result tables, figures, and LaTeX manuscript for the paper:

**From Normal Uniform Variance Mixtures to Triangular Variance Mixture Normal: Theory, Inference, Simulation, and Real-Data Applications**

Author: **Ushosree Raha**  
Affiliation: Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior

## Overview

This project studies two bounded normal variance-mixture distributions:

- **NUVM**: Normal Uniform Variance Mixture, where the latent variance follows a uniform distribution.
- **TVMN**: Triangular Variance Mixture Normal, where the latent variance follows a triangular distribution with lower, modal, and upper variance parameters.

The paper includes theory, density functions, moments, likelihood estimation, Monte Carlo simulation, bootstrap inference, Fisher information, real-data applications, and benchmark comparisons.

## Repository Structure

```text
NUVM/                         NUVM code, outputs, and figures
TVMN/                         TVMN code, outputs, and figures
paper/                        LaTeX manuscript and submission figures
paper/main.tex                Main LaTeX source file
paper/main.pdf                Compiled manuscript PDF
paper/figures/                Figures used in the manuscript
paper/supplementary_data/     Main CSV result tables used in the paper
```

## Main Results

The manuscript reports:

- NUVM density, CDF, survival, hazard, reverse hazard, and cumulative hazard functions.
- NUVM moments up to eighth order.
- TVMN semi-closed-form density using incomplete gamma functions.
- TVMN moments, kurtosis, characteristic function, and moment generating function.
- Maximum likelihood estimation for both models.
- Monte Carlo studies with RMSE and bias summaries.
- Bootstrap confidence intervals and Fisher-information-based uncertainty estimates.
- Real-data and benchmark comparisons against standard distributions.

## Requirements

The Python scripts were developed using standard scientific Python packages:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
```

Optional packages may be required for financial-data download scripts, depending on the script used.

## Reproducing Results

Run the main scripts from the repository root.

### NUVM

```bash
python NUVM/NUVM_Simulation_and_Real_Data.py --part simulation --replications 1000
python NUVM/NUVM_Simulation_and_Real_Data.py --part real --bootstrap-replications 1000
python NUVM/NUVM_Reviewer_Diagnostics.py --part all --replications 1000
```

### TVMN

```bash
python TVMN/03_TVMN_likelihood_mle.py
python TVMN/04_TVMN_empirical_studies.py --part monte-carlo --mc-replications 1000
python TVMN/04_TVMN_empirical_studies.py --part bootstrap --bootstrap-replications 1000
python TVMN/04_TVMN_empirical_studies.py --part real-data
python TVMN/04_TVMN_empirical_studies.py --part benchmark
```

## Manuscript

The manuscript source is located at:

```text
paper/main.tex
```

To compile locally, open `paper/main.tex` in TeXstudio and compile with `pdfLaTeX`.

## Notes

The empirical results are honest benchmark results. TVMN is not claimed to dominate all standard distributions. Its main advantage is interpretability when latent variance is believed to be bounded and one variance level is most plausible.

## Citation

If you use this work, please cite the manuscript once it is published or available as a preprint.

## License

Add a license before public release. Recommended: MIT License for code and CC BY 4.0 for manuscript materials.
