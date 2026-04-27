"""Step 4 — exploratory data analysis figures (species-only grouping)."""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import DATA, FIGURES, PROCESSED, SPECIES_COLOR, SPECIES_LABEL, log

sns.set_style("whitegrid")
PALETTE = {SPECIES_LABEL[k]: v for k, v in SPECIES_COLOR.items()}
ORDER = ["Corn", "Soybean"]


def load_metrics():
    df = pd.read_csv(DATA / "leaf_metrics.csv")
    df = df[df["seg_failed"] == False].copy()
    df["species_label"] = df["species"].map(SPECIES_LABEL)
    return df


def fig1_sample_overview(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["species_label"].value_counts().reindex(ORDER)
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[s] for s in counts.index], edgecolor="black")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(int(v)),
                ha="center", fontsize=11)
    ax.set_ylabel("Leaf sample count")
    ax.set_title("Figure 1. Sample counts by species")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig01_sample_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_leaf_area(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df, x="species_label", y="area_px", order=ORDER,
                   palette=PALETTE, inner="quartile", ax=ax, cut=0)
    sns.stripplot(data=df, x="species_label", y="area_px", order=ORDER,
                  color="black", size=3, alpha=0.4, jitter=0.2, ax=ax)
    ax.set_ylabel("Leaf area (pixels²)")
    ax.set_xlabel("Species")
    ax.set_title("Figure 2. Distribution of segmented leaf area extracted from each sample, grouped by species")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig02_leaf_area_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_greenness(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="species_label", y="greenness_idx", order=ORDER,
                palette=PALETTE, ax=ax)
    sns.stripplot(data=df, x="species_label", y="greenness_idx", order=ORDER,
                  color="black", size=3, alpha=0.5, jitter=0.2, ax=ax)
    ax.set_ylabel("Greenness index  G/(R+G+B)")
    ax.set_xlabel("Species")
    ax.set_title("Figure 3. Distribution of greenness index extracted from each leaf, grouped by species")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig03_greenness_index.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_lab_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    for sp in ["CN", "SB"]:
        sub = df[df["species"] == sp]
        ax.scatter(sub["mean_a"], sub["mean_b"], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.7, edgecolor="black", s=40)
        # 2-sigma ellipse
        if len(sub) > 2:
            xa, ya = sub["mean_a"].values, sub["mean_b"].values
            cov = np.cov(xa, ya)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 2 * np.sqrt(vals)
            ellipse = mpatches.Ellipse((xa.mean(), ya.mean()), w, h, angle=angle,
                                       edgecolor=SPECIES_COLOR[sp], facecolor="none",
                                       linewidth=2)
            ax.add_patch(ellipse)
    ax.set_xlabel("Mean a* (red-green)")
    ax.set_ylabel("Mean b* (blue-yellow)")
    ax.set_title("Figure 4. LAB colorspace coordinates (a* vs b*) extracted from each leaf")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig04_lab_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_texture_heatmap(df):
    """Pearson correlation of the curated feature set (excluding identifiers)."""
    feat_cols = df.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed"] if c in df.columns], errors="ignore")
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Figure 5. Pearson correlation matrix of the curated feature set")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig05_texture_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig6_leaf_panel(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, sp in enumerate(["CN", "SB"]):
        sub = df[df["species"] == sp].copy()
        med = sub["area_px"].median()
        sub["d"] = (sub["area_px"] - med).abs()
        rep = sub.sort_values("d").iloc[0]
        stem = Path(rep["filename"]).stem
        orig = cv2.imread(str(PROCESSED / f"{stem}.png"))
        mask = cv2.imread(str(PROCESSED / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        masked = orig.copy()
        masked[mask == 0] = 255
        axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{SPECIES_LABEL[sp]} — original")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{SPECIES_LABEL[sp]} — masked")
        axes[i, 1].axis("off")
    plt.suptitle("Figure 6. Representative leaf images and segmentation masks from the extracted dataset", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig06_leaf_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig7_pca(df):
    num_cols = df.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed"] if c in df.columns], errors="ignore")
    X = df[num_cols].fillna(0.0).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    mag = np.sqrt((loadings ** 2).sum(axis=1))
    top5 = np.argsort(mag)[::-1][:5]

    fig, ax = plt.subplots(figsize=(9, 7))
    for sp in ["CN", "SB"]:
        m = (df["species"].values == sp)
        ax.scatter(scores[m, 0], scores[m, 1], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.75, edgecolor="black", s=45)
    s = 1.2 * max(np.abs(scores).max(), 1) / max(np.abs(loadings).max(), 1e-9)
    for i in top5:
        ax.arrow(0, 0, loadings[i, 0] * s, loadings[i, 1] * s,
                 color="crimson", alpha=0.8, head_width=0.15, length_includes_head=True)
        ax.text(loadings[i, 0] * s * 1.08, loadings[i, 1] * s * 1.08,
                num_cols[i], color="crimson", fontsize=9)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("Figure 7. PCA biplot of the extracted leaf phenotypic feature set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig07_pca_biplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log("04_eda", "start")
    df = load_metrics()
    fig1_sample_overview(df)
    fig2_leaf_area(df)
    fig3_greenness(df)
    fig4_lab_scatter(df)
    fig5_texture_heatmap(df)
    fig6_leaf_panel(df)
    fig7_pca(df)
    log("04_eda", "wrote fig01..fig07")


if __name__ == "__main__":
    main()
