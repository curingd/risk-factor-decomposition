"""
Principal Component Analysis (PCA) of time-series risk factors
===============================================================

Purpose
-------
Run PCA on the set of time-series factor columns in ``factor_returns.csv``,
*excluding* the ``asset_returns`` series (the dependent/target series) and the
``date`` index. The goal is to understand the common latent structure driving
the 15 factors: how many independent sources of variation there are, how much
variance each explains, and how the original factors load onto them.

Why standardise first
---------------------
The factors are quoted in very different units and scales. For example ``VIX``
has a standard deviation of ~6.6 (index points) while ``slope_change`` has a
standard deviation of ~0.016. PCA decomposes variance, so on raw data the
large-variance columns (VIX, VIX_change, the credit spreads) would mechanically
dominate the leading components regardless of their statistical importance.

The standard remedy is to z-score each column (mean 0, unit variance) before
decomposing. This is equivalent to running PCA on the *correlation* matrix
rather than the *covariance* matrix, and it puts every factor on an equal
footing. We use ``StandardScaler`` for this.

Outputs
-------
1. Console / text report:
   - explained variance per component and cumulative variance
   - eigenvalues and the components retained under three common rules
   - the loadings (eigenvectors) matrix
2. Figures (PNG):
   - scree plot with cumulative variance
   - loadings heatmap (factor x component)
   - PC1-vs-PC2 loadings biplot
3. CSV artefacts:
   - explained variance table
   - loadings table
   - principal component scores (time series of each PC)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Paths resolve relative to this script's location, so it runs as-is after a
# clone with no absolute paths to edit.
REPO_ROOT = Path(__file__).resolve().parent.parent    # PCA/ -> repo root
INPUT_CSV = REPO_ROOT / "Excel" / "factor_returns.csv"
OUTPUT_DIR = Path(__file__).resolve().parent          # CSV artefacts -> PCA/ (beside this script)
IMAGE_DIR = REPO_ROOT / "Images"                      # figures -> Images/
EXCLUDE = ["date", "asset_returns"]   # not part of the factor block

plt.style.use("seaborn-v0_8-whitegrid")

# --------------------------------------------------------------------------- #
# 1. Load data and isolate the factor block
# --------------------------------------------------------------------------- #
df = pd.read_csv(INPUT_CSV)

# Parse the "Jun-13" style month labels into a proper datetime index. This is
# only used for plotting/exporting the PC score time series; it plays no part
# in the decomposition itself.
df["date"] = pd.to_datetime(df["date"], format="%b-%y")
df = df.sort_values("date").reset_index(drop=True)

# The factor matrix X: every numeric column except the excluded ones.
factor_names = [c for c in df.columns if c not in EXCLUDE]
X = df[factor_names].copy()

print(f"Observations (rows): {X.shape[0]}")
print(f"Factors (columns):   {X.shape[1]}")
print(f"Factors analysed:    {factor_names}")
assert X.isna().sum().sum() == 0, "Unexpected missing values in factor block"

# --------------------------------------------------------------------------- #
# 2. Standardise the factors (z-score -> PCA on the correlation matrix)
# --------------------------------------------------------------------------- #
scaler = StandardScaler()
X_std = scaler.fit_transform(X)   # shape (n_obs, n_factors), each col ~ N(0, 1)

# --------------------------------------------------------------------------- #
# 3. Fit PCA
# --------------------------------------------------------------------------- #
# With n_components=None, scikit-learn keeps all components, so we get the full
# spectrum of eigenvalues and can decide retention afterwards. PCA centres the
# data internally; we have also standardised, which is what we want here.
pca = PCA(n_components=None, svd_solver="full")
scores = pca.fit_transform(X_std)        # PC scores, shape (n_obs, n_factors)

# Explained variance.
#   explained_variance_      -> eigenvalues of the correlation matrix
#   explained_variance_ratio_-> fraction of total variance per component
eigenvalues = pca.explained_variance_
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

n_comp = len(evr)
pc_labels = [f"PC{i+1}" for i in range(n_comp)]

# --------------------------------------------------------------------------- #
# 4. Component-retention diagnostics
# --------------------------------------------------------------------------- #
# Three common, complementary rules of thumb:
#   (a) Kaiser criterion: keep components with eigenvalue > 1 (i.e. a PC that
#       explains more variance than a single standardised original factor).
#   (b) Cumulative variance: keep enough PCs to reach, say, 90% of variance.
#   (c) Scree/elbow: inspected visually in the scree plot.
kaiser_n = int(np.sum(eigenvalues > 1))
var_90_n = int(np.argmax(cum_evr >= 0.90) + 1)
var_80_n = int(np.argmax(cum_evr >= 0.80) + 1)

variance_table = pd.DataFrame(
    {
        "eigenvalue": eigenvalues,
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_evr,
    },
    index=pc_labels,
)

print("\n=== Explained variance ===")
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(variance_table)

print("\n=== Components to retain ===")
print(f"Kaiser (eigenvalue > 1):        {kaiser_n} components")
print(f">= 80% cumulative variance:     {var_80_n} components")
print(f">= 90% cumulative variance:     {var_90_n} components")

# --------------------------------------------------------------------------- #
# 5. Loadings (eigenvectors)
# --------------------------------------------------------------------------- #
# pca.components_ has shape (n_components, n_factors); row i is the eigenvector
# for PCi. We transpose so rows are the original factors and columns are PCs,
# which reads more naturally as "how does each factor load on each component".
loadings = pd.DataFrame(
    pca.components_.T,
    index=factor_names,
    columns=pc_labels,
)

print("\n=== Loadings (first 4 components) ===")
with pd.option_context("display.float_format", lambda v: f"{v:+0.3f}"):
    print(loadings.iloc[:, :4])

# For each of the leading components, list the factors with the largest
# absolute loading -- this is how you interpret what a component "means".
print("\n=== Dominant factors per leading component ===")
for pc in pc_labels[: max(kaiser_n, 3)]:
    top = loadings[pc].abs().sort_values(ascending=False).head(4).index
    desc = ", ".join(f"{f} ({loadings.loc[f, pc]:+.2f})" for f in top)
    print(f"{pc}: {desc}")

# --------------------------------------------------------------------------- #
# 6. Principal component score time series
# --------------------------------------------------------------------------- #
scores_df = pd.DataFrame(scores, columns=pc_labels)
scores_df.insert(0, "date", df["date"].values)

# --------------------------------------------------------------------------- #
# 7. Export tables
# --------------------------------------------------------------------------- #
variance_table.to_csv(f"{OUTPUT_DIR}/pca_explained_variance.csv")
loadings.to_csv(f"{OUTPUT_DIR}/pca_loadings.csv")
scores_df.to_csv(f"{OUTPUT_DIR}/pca_scores.csv", index=False)

# --------------------------------------------------------------------------- #
# 8. Figures
# --------------------------------------------------------------------------- #
# 8a. Scree plot + cumulative variance
fig, ax1 = plt.subplots(figsize=(9, 5))
x_pos = np.arange(1, n_comp + 1)
ax1.bar(x_pos, evr * 100, color="#4C72B0", alpha=0.85, label="Individual")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Variance explained (%)", color="#4C72B0")
ax1.set_xticks(x_pos)
ax1.tick_params(axis="y", labelcolor="#4C72B0")

ax2 = ax1.twinx()
ax2.plot(x_pos, cum_evr * 100, color="#C44E52", marker="o", label="Cumulative")
ax2.axhline(90, color="grey", linestyle="--", linewidth=1)
ax2.set_ylabel("Cumulative variance (%)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax2.set_ylim(0, 105)
ax1.set_title("PCA scree plot — factor returns")
fig.tight_layout()
fig.savefig(f"{IMAGE_DIR}/pca_scree.png", dpi=150)
plt.close(fig)

# 8b. Loadings heatmap (leading components only, for legibility)
n_show = max(kaiser_n, 5)
fig, ax = plt.subplots(figsize=(1.1 * n_show + 3, 0.45 * len(factor_names) + 2))
sns.heatmap(
    loadings.iloc[:, :n_show],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt="+.2f",
    linewidths=0.5,
    cbar_kws={"label": "Loading"},
    ax=ax,
)
ax.set_title(f"PCA loadings (PC1–PC{n_show})")
fig.tight_layout()
fig.savefig(f"{IMAGE_DIR}/pca_loadings_heatmap.png", dpi=150)
plt.close(fig)

# 8c. PC1 vs PC2 loadings biplot
fig, ax = plt.subplots(figsize=(8, 8))
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
for factor in factor_names:
    x, y = loadings.loc[factor, "PC1"], loadings.loc[factor, "PC2"]
    ax.arrow(0, 0, x, y, head_width=0.015, color="#4C72B0", alpha=0.7,
             length_includes_head=True)
    ax.text(x * 1.08, y * 1.08, factor, fontsize=8, ha="center", va="center")
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
ax.set_title("Factor loadings: PC1 vs PC2")
lim = np.abs(loadings[["PC1", "PC2"]].values).max() * 1.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{IMAGE_DIR}/pca_biplot.png", dpi=150)
plt.close(fig)

print("\nSaved: pca_explained_variance.csv, pca_loadings.csv, pca_scores.csv")
print("Saved: pca_scree.png, pca_loadings_heatmap.png, pca_biplot.png")
