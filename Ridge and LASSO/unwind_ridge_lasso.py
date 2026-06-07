"""
Un-normalising the optimal Ridge / Lasso fits to the raw factor series,
and post-selection OLS inference
======================================================================

Part 1 -- algebraic un-normalisation
------------------------------------
The penalised models were fit on standardised factors z_j = (x_j - mu_j)/sigma_j:
        yhat = b0_std + sum_j b_std_j * z_j
Substituting z_j and collecting terms gives the SAME fitted model written on the
raw series x_j:
        b_raw_j = b_std_j / sigma_j
        b0_raw  = b0_std - sum_j b_std_j * mu_j / sigma_j
        yhat    = b0_raw + sum_j b_raw_j * x_j
Predictions, residuals and in-sample R^2 are unchanged by this rewrite -- only
the units of the coefficients change. We verify that numerically.

Part 2 -- why there are no t-statistics for the penalised coefficients
----------------------------------------------------------------------
Ridge and Lasso are biased by construction, so the ordinary t = coef/SE does not
describe their sampling distribution; for Lasso the distribution is non-standard
because of the selection step. We therefore do NOT fabricate t-stats for the
shrunk coefficients. Instead we use Lasso as a variable SELECTOR (it retained 3
factors) and refit an ordinary unpenalised OLS on those 3 raw factors --
"post-Lasso OLS" -- which yields legitimate R^2, F and t-statistics to compare
with the original 15-factor OLS.

Caveat printed in the output: because the 3 factors were chosen using the same
data, the post-selection t-stats are optimistic (post-selection inference
problem). They are indicative, not exact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

pd.set_option("display.width", 200)
REPO_ROOT = Path(__file__).resolve().parent.parent    # "Ridge and LASSO/" -> repo root
INPUT_CSV = REPO_ROOT / "Excel" / "factor_returns.csv"
OUT = Path(__file__).resolve().parent                 # outputs -> "Ridge and LASSO/"
EXCLUDE = ["date", "asset_returns"]
RIDGE_ALPHA = 200.9      # CV-selected earlier
LASSO_ALPHA = 0.3054     # CV-selected earlier

# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], format="%b-%y")
df = df.sort_values("date").reset_index(drop=True)
factors = [c for c in df.columns if c not in EXCLUDE]
X = df[factors].to_numpy(float)
y = df["asset_returns"].to_numpy(float)

# Standardisation constants (full sample, ddof=0 to match StandardScaler).
scaler = StandardScaler().fit(X)
mu, sigma = scaler.mean_, scaler.scale_
Z = scaler.transform(X)

# --------------------------------------------------------------------------- #
# Part 1 -- fit penalised models, then un-normalise
# --------------------------------------------------------------------------- #
def unwind(model):
    """Return (intercept_raw, coef_raw) on the original factor units."""
    b_std = model.coef_
    b0_std = model.intercept_
    b_raw = b_std / sigma
    b0_raw = b0_std - np.sum(b_std * mu / sigma)
    return b0_raw, b_raw

ridge = Ridge(alpha=RIDGE_ALPHA).fit(Z, y)
lasso = Lasso(alpha=LASSO_ALPHA, max_iter=500000, tol=1e-3).fit(Z, y)

ridge_b0, ridge_b = unwind(ridge)
lasso_b0, lasso_b = unwind(lasso)

raw_tbl = pd.DataFrame(
    {"Ridge (raw)": ridge_b, "Lasso (raw)": lasso_b}, index=factors
)
print("=" * 78)
print("PART 1 -- COEFFICIENTS ON RAW FACTOR SERIES (un-normalised)")
print("=" * 78)
with pd.option_context("display.float_format", lambda v: f"{v:+0.6f}"):
    print(raw_tbl)
print(f"\nRidge intercept (raw): {ridge_b0:+.6f}")
print(f"Lasso intercept (raw): {lasso_b0:+.6f}")

# Verify the rewrite reproduces the standardised model exactly.
def r2(yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

ridge_raw_pred = ridge_b0 + X @ ridge_b
lasso_raw_pred = lasso_b0 + X @ lasso_b
print("\nEquivalence check (raw rewrite vs standardised fit)")
print(f"  Ridge: max|pred diff| = {np.max(np.abs(ridge_raw_pred - ridge.predict(Z))):.3e}, "
      f"R^2 = {r2(ridge_raw_pred):.4f}")
print(f"  Lasso: max|pred diff| = {np.max(np.abs(lasso_raw_pred - lasso.predict(Z))):.3e}, "
      f"R^2 = {r2(lasso_raw_pred):.4f}")

# --------------------------------------------------------------------------- #
# Part 2 -- post-Lasso OLS on the selected raw factors
# --------------------------------------------------------------------------- #
selected = [f for f, c in zip(factors, lasso.coef_) if abs(c) > 1e-8]
print("\n" + "=" * 78)
print(f"PART 2 -- POST-LASSO OLS on {len(selected)} selected raw factors")
print(f"Selected: {selected}")
print("=" * 78)

Xsel = sm.add_constant(df[selected].to_numpy(float))
post = sm.OLS(y, Xsel).fit()
print(post.summary(xname=["const"] + selected, yname="asset_returns"))

print("\nHeadline")
print(f"  R^2 = {post.rsquared:.4f}   adj R^2 = {post.rsquared_adj:.4f}")
print(f"  F = {post.fvalue:.2f}  (df {int(post.df_model)}, {int(post.df_resid)}),  "
      f"p(F) = {post.f_pvalue:.3e}")

# --------------------------------------------------------------------------- #
# Part 3 -- original 15-factor OLS, for side-by-side comparison
# --------------------------------------------------------------------------- #
full = sm.OLS(y, sm.add_constant(X)).fit()
print("\n" + "=" * 78)
print("PART 3 -- COMPARISON WITH ORIGINAL 15-FACTOR OLS")
print("=" * 78)
cmp = pd.DataFrame(
    {
        "model": ["Original OLS (15 factors)", "Post-Lasso OLS (3 factors)"],
        "R^2": [full.rsquared, post.rsquared],
        "adj_R^2": [full.rsquared_adj, post.rsquared_adj],
        "F": [full.fvalue, post.fvalue],
        "n_params": [int(full.df_model), int(post.df_model)],
    }
)
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(cmp.to_string(index=False))

# t-stats of the 3 factors in both models, side by side
common = pd.DataFrame({
    "t (post-Lasso OLS)": pd.Series(post.tvalues[1:], index=selected),
    "t (original 15-factor OLS)": pd.Series(
        np.asarray(full.tvalues)[1:], index=factors).loc[selected],
})
print("\nt-statistics for the 3 selected factors")
with pd.option_context("display.float_format", lambda v: f"{v:+0.3f}"):
    print(common)

print("\nNOTE: post-selection t-stats are optimistic -- the 3 factors were chosen")
print("using the same data, so these overstate significance. Indicative only.")

# Export
raw_tbl.assign(**{"Ridge_intercept": ridge_b0, "Lasso_intercept": lasso_b0}
               ).to_csv(f"{OUT}/ridge_lasso_raw_coefficients.csv")
with open(f"{OUT}/post_lasso_ols_summary.txt", "w") as fh:
    fh.write(post.summary(xname=["const"] + selected, yname="asset_returns").as_text())
print("\nSaved: ridge_lasso_raw_coefficients.csv, post_lasso_ols_summary.txt")
