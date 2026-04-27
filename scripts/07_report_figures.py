"""Step 7 — regenerate publication-quality figures (PNG + SVG)."""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import DATA, FIGURES, PROCESSED, SPECIES_COLOR, SPECIES_LABEL, log

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
sns.set_style("whitegrid")

PALETTE = {SPECIES_LABEL[k]: v for k, v in SPECIES_COLOR.items()}
ORDER = ["Corn", "Soybean"]


def save_both(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log("07_report_figures", "start")
    df = pd.read_csv(DATA / "leaf_metrics.csv")
    df = df[df["seg_failed"] == False].copy()
    df["species_label"] = df["species"].map(SPECIES_LABEL)

    # --- report_fig02 leaf area ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df, x="species_label", y="area_px", order=ORDER,
                   palette=PALETTE, inner="quartile", ax=ax, cut=0)
    sns.stripplot(data=df, x="species_label", y="area_px", order=ORDER,
                  color="black", size=3, alpha=0.4, jitter=0.2, ax=ax)
    ax.set_ylabel("Leaf area (pixels²)"); ax.set_xlabel("Species")
    ax.set_title("Distribution of extracted leaf area, grouped by species")
    save_both(fig, "report_fig02_leaf_area")

    # --- report_fig03 greenness ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="species_label", y="greenness_idx", order=ORDER,
                palette=PALETTE, ax=ax)
    sns.stripplot(data=df, x="species_label", y="greenness_idx", order=ORDER,
                  color="black", size=3, alpha=0.5, jitter=0.2, ax=ax)
    ax.set_ylabel("Greenness index G/(R+G+B)"); ax.set_xlabel("Species")
    ax.set_title("Distribution of extracted greenness index, grouped by species")
    save_both(fig, "report_fig03_greenness")

    # --- report_fig04 LAB ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for sp in ["CN", "SB"]:
        sub = df[df.species == sp]
        ax.scatter(sub["mean_a"], sub["mean_b"], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.75, edgecolor="black", s=40)
        if len(sub) > 2:
            xa, ya = sub["mean_a"].values, sub["mean_b"].values
            cov = np.cov(xa, ya); vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 2 * np.sqrt(vals)
            ax.add_patch(mpatches.Ellipse((xa.mean(), ya.mean()), w, h, angle=angle,
                                          edgecolor=SPECIES_COLOR[sp], facecolor="none", lw=2))
    ax.set_xlabel("Mean a*"); ax.set_ylabel("Mean b*")
    ax.set_title("Extracted LAB colorspace coordinates (a* vs b*)"); ax.legend()
    save_both(fig, "report_fig04_lab_scatter")

    # --- report_fig06 panel ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, sp in enumerate(["CN", "SB"]):
        sub = df[df.species == sp].copy()
        med = sub["area_px"].median()
        sub["d"] = (sub["area_px"] - med).abs()
        rep = sub.sort_values("d").iloc[0]
        stem = Path(rep["filename"]).stem
        orig = cv2.imread(str(PROCESSED / f"{stem}.png"))
        mask = cv2.imread(str(PROCESSED / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        masked = orig.copy(); masked[mask == 0] = 255
        axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{SPECIES_LABEL[sp]} — original"); axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{SPECIES_LABEL[sp]} — masked"); axes[i, 1].axis("off")
    plt.suptitle("Representative leaves from the extracted dataset", fontsize=13)
    plt.tight_layout()
    save_both(fig, "report_fig06_panel")

    # --- report_fig07 PCA ---
    num_cols = df.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed"] if c in df.columns], errors="ignore")
    X = StandardScaler().fit_transform(df[num_cols].fillna(0.0).values)
    pca = PCA(n_components=2); scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    mag = np.sqrt((loadings ** 2).sum(axis=1))
    top5 = np.argsort(mag)[::-1][:5]
    fig, ax = plt.subplots(figsize=(9, 7))
    for sp in ["CN", "SB"]:
        mk = df["species"].values == sp
        ax.scatter(scores[mk, 0], scores[mk, 1], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.75, edgecolor="black", s=45)
    s = 1.2 * max(np.abs(scores).max(), 1) / max(np.abs(loadings).max(), 1e-9)
    for i in top5:
        ax.arrow(0, 0, loadings[i, 0] * s, loadings[i, 1] * s,
                 color="crimson", alpha=0.8, head_width=0.15, length_includes_head=True)
        ax.text(loadings[i, 0] * s * 1.08, loadings[i, 1] * s * 1.08,
                num_cols[i], color="crimson", fontsize=9)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA biplot of the extracted leaf phenotypic feature set"); ax.legend()
    save_both(fig, "report_fig07_pca")

    # --- report_qc1 outliers ---
    flags = pd.read_csv(DATA / "qc_flags_per_image.csv")
    m = df.merge(flags, on=["filename", "sample_id", "species"], how="left")
    color = np.where(m["outlier_severe"], "red",
                     np.where(m["outlier_mild"], "orange", "steelblue"))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m["area_px"], m["greenness_idx"], c=color, alpha=0.75,
               edgecolor="black", s=40)
    ax.set_xlabel("Leaf area (px²)"); ax.set_ylabel("Greenness index")
    ax.set_title("Outlier detection")
    ax.legend(handles=[mpatches.Patch(color="steelblue", label="normal"),
                       mpatches.Patch(color="orange", label="mild outlier"),
                       mpatches.Patch(color="red", label="extreme outlier")])
    save_both(fig, "report_qc1_outliers")

    # --- report_qc2 exposure ---
    L = df["mean_L"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(L, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(12, color="red", linestyle="--"); ax.axvline(85, color="red", linestyle="--")
    ax.axvspan(0, 12, alpha=0.2, color="red"); ax.axvspan(85, 100, alpha=0.2, color="red")
    ax.set_xlabel("Mean L* (CIE 0-100)"); ax.set_ylabel("Image count")
    ax.set_title("Exposure distribution")
    save_both(fig, "report_qc2_exposure")

    log("07_report_figures", "wrote report_fig02..07 and report_qc1..2 (png+svg)")


if __name__ == "__main__":
    main()
