# Risk Factor Decomposition

This project takes a simulation of asset returns from a typical immature DB 
pension plan in 2011, over the period from June 2013 to December 2025 and 
decomposes the returns into various risk factors using OLS and LASSO regression, 
with principal components analysis (PCA) also performed in Python for a more 
detailed analysis of explanatory variable variances.

The return series is generated using index proxies for various asset classes 
and the typical asset allocation taken from the PPF Purple Book 2011. 

For the full DB pension plan funding simulator project that the series was drawn from, 
visit my GitHub page: `https://github.com/curingd/db-pension-plan-funding-simulator`

The key findings are that returns are mainly explained by exposures to credit 
and changes in inflation using the OLS model, or credit and a negative linear 
relationship with equity using the LASSO model. These results are not contradictory 
as OLS regressions of S&P 500 returns over high-yield bonds on credit factors and 
inflation changes on credit and S&P 500 factors show high significance, so whether 
the factors are framed as credit, equity, or inflation exposures over the 
2013-25 period are a matter of preference.

A full write-up of the methodology and findings can be found in [`Analysis/analysis.pdf`](Analysis/analysis.pdf)

---

## Data

Raw inputs live in [`Data/`](Data/) and are assembled into the modelling panel
[`Excel/factor_returns.csv`](Excel/factor_returns.csv):

- **`SLXX`** — sterling investment-grade corporate bond series (CSV + XLSX).
- **`VIX`** — equity-volatility index, used to construct `VIX` and `VIX_change`.
- **`OIS month end data` (2009–2015, 2016–2024)** — overnight index swap / rate
  data underpinning the SONIA, spread, and curve factors.
- **`ie_data.xlsx`** — Shiller-style equity dataset behind the S&P 500 equity
  risk premium (`S&P_500_ERP`) factor.

Baseline OLS models were first built independently in Excel
([`Excel/risk_factor_decomposition_ols.xlsx`](Excel/risk_factor_decomposition_ols.xlsx))
before being reproduced and extended in Python.

---

## Repository structure

```
risk-factor-decomposition/
├── Analysis/                       # Narrative write-up of the full study
│   ├── analysis.pdf
│   └── analysis.docx
├── Data/                           # Raw source data
│   ├── SLXX.csv / SLXX.xlsx
│   ├── VIX.csv / VIX.xlsx
│   ├── OIS month end data_2009 to 2015.xlsx
│   ├── OIS month end data_2016 to 2024.xlsx
│   └── ie_data.xlsx
├── Excel/
│   ├── factor_returns.csv          # Assembled panel: date, asset_returns, 15 factors
│   └── risk_factor_decomposition_ols.xlsx   # Baseline OLS in Excel
├── PCA/
│   ├── pca_factor_analysis.py      # PCA on the standardised factor block
│   ├── pca_explained_variance.csv
│   ├── pca_loadings.csv
│   └── pca_scores.csv
├── Ridge and LASSO/
│   ├── ridge_lasso.py              # Ridge & Lasso with time-series CV
│   ├── unwind_ridge_lasso.py       # Un-normalisation + post-Lasso OLS inference
│   ├── ridge_lasso_coefficients.csv
│   ├── ridge_lasso_raw_coefficients.csv
│   ├── ridge_lasso_fit.csv
│   └── post_lasso_ols_summary.txt
├── Model Comparison/
│   ├── model_fit_table.py          # R², adj-R², AIC, BIC across specifications
│   └── model_fit_table.csv
├── Images/                         # All diagnostic figures (scree, paths, Q-Q, etc.)
├── LICENSE                         # GPL-3.0
└── README.md
```
---

## ⚠️ A note on inference

The post-Lasso OLS t-statistics are **not valid for formal inference**: the same
data was used to select the three factors and to test them, so the reported
standard errors and p-values are optimistic (the classic post-selection
inference problem). They are indicative, not exact — useful for comparison
against the full-sample OLS, not for hypothesis testing as-is.

---

## Reproducing the analysis

### Requirements

Python 3.9+ with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### Running

Each stage is a standalone script. Note that the scripts read the panel from a
hard-coded `INPUT_CSV` path and write figures/CSVs to an output directory —
point these at [`Excel/factor_returns.csv`](Excel/factor_returns.csv) and a local
folder before running:

```bash
python "PCA/pca_factor_analysis.py"
python "Ridge and LASSO/ridge_lasso.py"
python "Ridge and LASSO/unwind_ridge_lasso.py"
python "Model Comparison/model_fit_table.py"
```

---

## License

Released under the [GNU General Public License v3.0](LICENSE).
