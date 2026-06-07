"""
model_fit_table.py
------------------
Compute and tabulate in-sample model-fit statistics (R^2, adjusted R^2,
AIC, BIC) for three competing specifications of asset_returns, and render
the comparison as a clean table image.

The three models
================
  1. OLS (credit, dinflation, dvix)       -- the manually-selected 4-factor
        OLS: IG_gilt_spread, HY_IG_spread, inflation_change, VIX_change.
  2. Lasso (credit, equity, dvix)         -- the CV-optimal Lasso (alpha =
        0.3054) SHRUNK fit on the 3 selected factors: IG_gilt_spread,
        S&P_500_HY_spread, VIX_change.  Coefficients are biased toward zero.
  3. Post-Lasso OLS (credit, equity, dvix)-- plain OLS refit (unbiased) on
        the same 3 Lasso-selected factors.

METHODOLOGICAL NOTES
====================
* INVARIANCE TO UNWINDING.  All four statistics are functions of the fitted
  values y_hat (via RSS) and an integer parameter count; they are invariant
  to the standardisation-unwinding b_raw = diag(1/sigma) . b_std.  The Lasso
  row is therefore scored directly from the supplied raw (unnormalised)
  coefficients on the raw factor matrix -- this reproduces the standardised
  fit exactly (verified separately to < 1e-13).

* DEGREES OF FREEDOM ARE UNAMBIGUOUS HERE.  Unlike Ridge (which shrinks but
  zeroes nothing, forcing a fractional effective-df = trace of the hat
  matrix), all three models below have an integer slope count: OLS = 4,
  Lasso = 3 non-zero, Post-Lasso OLS = 3.  No effective-df convention is
  needed.

* AIC / BIC CONVENTION (held identical across models for comparability).
  Gaussian log-likelihood with the ML variance estimate sigma^2 = RSS/n:
      logL = -0.5 * n * ( ln(2*pi) + ln(RSS/n) + 1 )
  Parameter count k = (slopes + intercept) + 1, the final +1 for the
  estimated error variance.  Then
      AIC = 2k - 2*logL ,   BIC = ln(n)*k - 2*logL .

* INTERPRETATION CAVEAT.  These are full-sample, in-sample statistics --
  appropriate for an ex-post variance decomposition (the project's purpose),
  where in-sample fit is the object of interest rather than forward-looking
  prediction.  The Lasso row's lower R^2 is largely a shrinkage artefact;
  the Post-Lasso OLS row removes that bias and is the like-for-like
  comparator for the OLS specification.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression

UPLOADS = "/mnt/user-data/uploads/"
OUT = "/mnt/user-data/outputs/"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
df = pd.read_csv(UPLOADS + "factor_returns.csv")
df["date"] = pd.to_datetime(df["date"], format="%b-%y")
df = df.sort_values("date").reset_index(drop=True)

y = df["asset_returns"].to_numpy(float)
n = len(y)
TSS = float(((y - y.mean()) ** 2).sum())


# --------------------------------------------------------------------------- #
# Statistics helper
# --------------------------------------------------------------------------- #
def fit_statistics(rss, n_slopes):
    """Return (R^2, adjusted R^2, AIC, BIC) from a residual sum of squares.

    Parameters
    ----------
    rss : float
        In-sample residual sum of squares of the fitted model.
    n_slopes : int
        Number of estimated slope coefficients (intercept excluded).

    Notes
    -----
    Adjusted R^2 uses residual df = n - n_slopes - 1.  AIC/BIC use the
    Gaussian ML log-likelihood and parameter count k = n_slopes + 2
    (slopes + intercept + error variance); see module docstring.
    """
    r2 = 1.0 - rss / TSS
    resid_df = n - n_slopes - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / resid_df
    log_lik = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)
    k = n_slopes + 2
    aic = 2 * k - 2 * log_lik
    bic = np.log(n) * k - 2 * log_lik
    return r2, adj_r2, aic, bic


def ols_rss(cols):
    """Fit plain OLS of asset_returns on `cols` and return its in-sample RSS."""
    x = df[cols].to_numpy(float)
    model = LinearRegression().fit(x, y)
    return float(((y - model.predict(x)) ** 2).sum())


# --------------------------------------------------------------------------- #
# Model 1: manually-selected 4-factor OLS
# --------------------------------------------------------------------------- #
ols_cols = ["IG_gilt_spread", "HY_IG_spread", "inflation_change", "VIX_change"]
rss_ols = ols_rss(ols_cols)
stat_ols = fit_statistics(rss_ols, n_slopes=4)

# --------------------------------------------------------------------------- #
# Model 2: CV-optimal Lasso, scored from the supplied raw (unnormalised)
#          coefficients -- the shrunk fit.
# --------------------------------------------------------------------------- #
factor_names = [c for c in df.columns if c not in ("date", "asset_returns")]
rl = pd.read_csv(UPLOADS + "ridge_lasso_raw_coefficients.csv", index_col=0)
b_lasso = rl.loc[factor_names, "Lasso (raw)"].to_numpy(float)
ic_lasso = float(rl["Lasso_intercept"].iloc[0])
yhat_lasso = ic_lasso + df[factor_names].to_numpy(float) @ b_lasso
rss_lasso = float(((y - yhat_lasso) ** 2).sum())
n_nonzero = int((np.abs(b_lasso) > 1e-12).sum())          # = 3
stat_lasso = fit_statistics(rss_lasso, n_slopes=n_nonzero)

# --------------------------------------------------------------------------- #
# Model 3: post-Lasso OLS -- unbiased refit on the 3 selected factors.
# --------------------------------------------------------------------------- #
post_cols = ["IG_gilt_spread", "S&P_500_HY_spread", "VIX_change"]
rss_post = ols_rss(post_cols)
stat_post = fit_statistics(rss_post, n_slopes=3)

# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
rows = [
    ("OLS (credit, dinflation, dvix)", stat_ols),
    ("Lasso (credit, equity, dvix)", stat_lasso),
    ("Post-Lasso OLS (credit, equity, dvix)", stat_post),
]
table = pd.DataFrame(
    [(name, *s) for name, s in rows],
    columns=["Model", "R2", "Adj_R2", "AIC", "BIC"],
)
table.to_csv(OUT + "model_fit_table.csv", index=False)
print(table.to_string(index=False))


# --------------------------------------------------------------------------- #
# Render clean table image
# --------------------------------------------------------------------------- #
NAVY = "#1F3864"
ZEBRA = "#F2F5FA"
GRID = "#BFBFBF"
INK = "#1A1A1A"

headers = ["Model", "$R^2$", "Adjusted $R^2$", "AIC", "BIC"]
cell_text = [
    [name, f"{r2:.4f}", f"{adj:.4f}", f"{aic:.2f}", f"{bic:.2f}"]
    for name, (r2, adj, aic, bic) in rows
]

# geometry (axes coordinates 0..1)
col_x = [0.015, 0.46, 0.62, 0.79, 0.92]   # left edge of text per column
col_align = ["left", "center", "center", "center", "center"]
n_rows = len(cell_text)
row_h = 1.0 / (n_rows + 1)

fig, ax = plt.subplots(figsize=(9.4, 2.35), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# header band
ax.add_patch(Rectangle((0, 1 - row_h), 1, row_h, facecolor=NAVY,
                        edgecolor="none", zorder=1))
for x, h, al in zip(col_x, headers, col_align):
    hx = x if al == "left" else (x if al != "center" else x)
    ax.text(hx, 1 - row_h / 2, h, ha=al, va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)

# body rows
for i, cells in enumerate(cell_text):
    y_top = 1 - (i + 2) * row_h
    if i % 2 == 1:
        ax.add_patch(Rectangle((0, y_top), 1, row_h, facecolor=ZEBRA,
                               edgecolor="none", zorder=1))
    for x, txt, al in zip(col_x, cells, col_align):
        weight = "bold" if al == "left" else "normal"
        ax.text(x, y_top + row_h / 2, txt, ha=al, va="center",
                color=INK, fontsize=10.5, fontweight=weight, zorder=3)

# horizontal rules
for i in range(n_rows + 2):
    yy = 1 - i * row_h
    lw = 1.4 if i in (0, 1, n_rows + 1) else 0.6
    ax.plot([0, 1], [yy, yy], color=GRID if i not in (1,) else NAVY,
            lw=lw, zorder=2)

ax.set_title("Risk-factor regressions: in-sample model fit",
             fontsize=12.5, fontweight="bold", color=NAVY, pad=10, loc="left")
fig.text(0.015, -0.02,
         "n = 144.  AIC/BIC: Gaussian ML, k = slopes + intercept + variance "
         "(identical convention across models). Lasso row = shrunk fit; "
         "Post-Lasso OLS = unbiased refit on the same three factors.",
         fontsize=6.6, color="#666666", ha="left")

plt.tight_layout()
fig.savefig(OUT + "model_fit_table.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("\nSaved: model_fit_table.png, model_fit_table.csv")
