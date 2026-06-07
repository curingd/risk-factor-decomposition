"""
Ridge and Lasso regression of asset_returns on the 15 factors
=============================================================

Why standardise (and why it matters more here than for OLS)
-----------------------------------------------------------
Ridge and Lasso add a penalty on the coefficient magnitudes:
    Ridge:  min  ||y - Xb||^2 + alpha * sum(b_j^2)
    Lasso:  min  (1/2n)||y - Xb||^2 + alpha * sum(|b_j|)
Because the penalty treats every coefficient on the same footing, a factor's
penalty depends on its units. VIX (sd ~6.6) and slope_change (sd ~0.016) would
be penalised utterly differently on raw data. So we z-score every factor first;
the coefficients below are therefore the effect of a ONE-STANDARD-DEVIATION move
in each factor, which also makes them directly comparable in magnitude.
y (asset_returns) is left in its original units -- the intercept is never
penalised (sklearn fits it separately on the centred data).

Choosing the penalty strength alpha
------------------------------------
alpha is selected by cross-validation. Because this is a monthly time series,
we use TimeSeriesSplit (forward-chaining: always train on the past, validate on
the future) rather than random k-fold, which would let the model peek ahead.
Scaling is done INSIDE each fold via a Pipeline, so the validation folds never
inform the mean/sd used to scale the training folds (no leakage).

Outputs
-------
1. CV-selected alpha for ridge and lasso.
2. Coefficient table: OLS vs Ridge vs Lasso (standardised units), plus which
   factors Lasso drives exactly to zero.
3. In-sample and cross-validated R^2 for all three models.
4. Figures: ridge & lasso coefficient paths vs alpha, CV-error curves, and a
   side-by-side coefficient comparison bar chart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.width", 200)
INPUT_CSV = "/mnt/user-data/uploads/factor_returns.csv"
OUT = "/home/claude"
EXCLUDE = ["date", "asset_returns"]

# --------------------------------------------------------------------------- #
# 1. Load and split into design matrix X and target y
# --------------------------------------------------------------------------- #
df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], format="%b-%y")
df = df.sort_values("date").reset_index(drop=True)   # chronological for TS-CV

factor_names = [c for c in df.columns if c not in EXCLUDE]
X = df[factor_names].to_numpy(float)
y = df["asset_returns"].to_numpy(float)
n, p = X.shape
print(f"n = {n} monthly observations, p = {p} factors")

# Time-series cross-validation splitter (5 expanding-window folds).
tscv = TimeSeriesSplit(n_splits=5)

# --------------------------------------------------------------------------- #
# 2. Select alpha by time-series CV (scaling inside each fold via Pipeline)
# --------------------------------------------------------------------------- #
ridge_alphas = np.logspace(-2, 4, 100)
lasso_alphas = np.logspace(-4, 1, 100)

ridge_grid = GridSearchCV(
    Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
    {"m__alpha": ridge_alphas}, cv=tscv, scoring="neg_mean_squared_error",
).fit(X, y)
lasso_grid = GridSearchCV(
    Pipeline([("sc", StandardScaler()), ("m", Lasso(max_iter=500000, tol=1e-3))]),
    {"m__alpha": lasso_alphas}, cv=tscv, scoring="neg_mean_squared_error",
).fit(X, y)

ridge_alpha = ridge_grid.best_params_["m__alpha"]
lasso_alpha = lasso_grid.best_params_["m__alpha"]
print(f"\nCV-selected alpha  ridge: {ridge_alpha:.4g}")
print(f"CV-selected alpha  lasso: {lasso_alpha:.4g}")

# best_estimator_ is refit on ALL data, so its scaler holds the full-sample
# mean / sd -- exactly the mu, sigma we want to report.
mu = ridge_grid.best_estimator_.named_steps["sc"].mean_
sigma = ridge_grid.best_estimator_.named_steps["sc"].scale_
print("\nStandardisation constants (full sample)")
with pd.option_context("display.float_format", lambda v: f"{v:0.6f}"):
    print(pd.DataFrame({"mean": mu, "std (ddof=0)": sigma}, index=factor_names))

# --------------------------------------------------------------------------- #
# 3. Coefficients (in standardised-factor units) for OLS / Ridge / Lasso
# --------------------------------------------------------------------------- #
ridge_coef = ridge_grid.best_estimator_.named_steps["m"].coef_
lasso_coef = lasso_grid.best_estimator_.named_steps["m"].coef_

ols_pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]).fit(X, y)
ols_coef = ols_pipe.named_steps["m"].coef_

coef_tbl = pd.DataFrame(
    {"OLS": ols_coef, "Ridge": ridge_coef, "Lasso": lasso_coef}, index=factor_names
)
print("\nCoefficients on STANDARDISED factors (effect of a 1-sd move)")
with pd.option_context("display.float_format", lambda v: f"{v:+0.4f}"):
    print(coef_tbl)

dropped = [f for f, c in zip(factor_names, lasso_coef) if abs(c) < 1e-8]
print(f"\nFactors zeroed out by Lasso ({len(dropped)}): {dropped}")

# --------------------------------------------------------------------------- #
# 4. Fit quality: in-sample R^2 and cross-validated R^2
# --------------------------------------------------------------------------- #
def scores(estimator):
    insample = estimator.fit(X, y).score(X, y)
    cv = cross_val_score(estimator, X, y, cv=tscv, scoring="r2").mean()
    return insample, cv

models = {
    "OLS": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=ridge_alpha))]),
    "Lasso": Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=lasso_alpha, max_iter=500000, tol=1e-3))]),
}
fit_tbl = pd.DataFrame(
    {name: scores(est) for name, est in models.items()},
    index=["in_sample_R2", "cv_R2 (TimeSeriesSplit)"],
).T
print("\nFit quality")
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(fit_tbl)

# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
Xs = StandardScaler().fit_transform(X)   # full-sample standardised, for paths

# 5a/5b. Coefficient paths
def coef_path(Model, alphas, **kw):
    return np.array([Model(alpha=a, **kw).fit(Xs, y).coef_ for a in alphas])

ridge_path = coef_path(Ridge, ridge_alphas)
lasso_path = coef_path(Lasso, lasso_alphas, max_iter=500000, tol=1e-3)

for path, alphas, sel, title, fname in [
    (ridge_path, ridge_alphas, ridge_alpha, "Ridge coefficient paths", "ridge_path.png"),
    (lasso_path, lasso_alphas, lasso_alpha, "Lasso coefficient paths", "lasso_path.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, name in enumerate(factor_names):
        ax.plot(alphas, path[:, j], label=name, linewidth=1.3)
    ax.axvline(sel, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (penalty strength)")
    ax.set_ylabel("standardised coefficient")
    ax.set_title(f"{title}  (dashed = CV-selected alpha = {sel:.3g})")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=150)
    plt.close(fig)

# 5c. CV-error curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, grid, alphas, sel, name in [
    (axes[0], ridge_grid, ridge_alphas, ridge_alpha, "Ridge"),
    (axes[1], lasso_grid, lasso_alphas, lasso_alpha, "Lasso"),
]:
    mse = -grid.cv_results_["mean_test_score"]
    ax.plot(alphas, mse, color="#4C72B0")
    ax.axvline(sel, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CV mean squared error")
    ax.set_title(f"{name}: CV error (min at alpha={sel:.3g})")
fig.tight_layout()
fig.savefig(f"{OUT}/cv_error_curves.png", dpi=150)
plt.close(fig)

# 5d. Coefficient comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(np.abs(ridge_coef))[::-1]
ypos = np.arange(len(factor_names))
w = 0.27
ax.barh(ypos - w, ols_coef[order], w, label="OLS", color="#BBBBBB")
ax.barh(ypos, ridge_coef[order], w, label="Ridge", color="#4C72B0")
ax.barh(ypos + w, lasso_coef[order], w, label="Lasso", color="#C44E52")
ax.set_yticks(ypos)
ax.set_yticklabels(np.array(factor_names)[order])
ax.invert_yaxis()
ax.axvline(0, color="grey", linewidth=0.6)
ax.set_xlabel("standardised coefficient")
ax.set_title("Coefficients: OLS vs Ridge vs Lasso")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/coef_comparison.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 6. Export tables
# --------------------------------------------------------------------------- #
coef_tbl.to_csv(f"{OUT}/ridge_lasso_coefficients.csv")
fit_tbl.to_csv(f"{OUT}/ridge_lasso_fit.csv")
print("\nSaved: ridge_path.png, lasso_path.png, cv_error_curves.png,")
print("       coef_comparison.png, ridge_lasso_coefficients.csv, ridge_lasso_fit.csv")
