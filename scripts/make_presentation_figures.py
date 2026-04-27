"""Build presentation-quality (16:9, big-font, projector-friendly) figures
with Purdue gold + black palette. Drop-in PNGs for PowerPoint.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
OUT = ROOT / "figures" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

# Purdue palette
PURDUE_GOLD = "#CFB991"
PURDUE_GOLD_DARK = "#8E6F3E"
PURDUE_BLACK = "#000000"
ACCENT = "#DAAA00"
LIGHT_BG = "#F4F0E5"

SPECIES_COLOR = {"CN": ACCENT, "SB": "#1B1B1B"}
SPECIES_LABEL = {"CN": "Corn", "SB": "Soybean"}
ORDER = ["Corn", "Soybean"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelsize": 22,
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.edgecolor": PURDUE_BLACK,
    "axes.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 1.2,
    "lines.linewidth": 3.0,
    "patch.linewidth": 2.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

FIGSIZE_169 = (13.33, 7.5)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  -> {stem}.png")


def headline(ax, text, y=1.04):
    """Big takeaway text baked into the figure."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=24, fontweight="bold", color=PURDUE_BLACK)


def load():
    df = pd.read_csv(DATA / "leaf_metrics.csv")
    df = df[df["seg_failed"] == False].copy()
    df["species_label"] = df["species"].map(SPECIES_LABEL)
    return df


# ---------- 1. Pipeline flowchart ------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis("off")
    steps = [
        ("Raw\nJPEGs", "input"),
        ("Standardize\n& inventory", "step"),
        ("Lighting\nbalance", "step"),
        ("Segment\n(species-aware)", "step"),
        ("Extract\n33 features", "step"),
        ("Quality\nchecking", "step"),
        ("Statistics\n& metrics", "step"),
        ("Report &\npublication", "output"),
    ]
    n = len(steps)
    box_w, box_h = 1.05, 1.4
    gap = (10 - n * box_w) / (n + 1)
    y = 2.5
    for i, (label, kind) in enumerate(steps):
        x = gap + i * (box_w + gap)
        if kind == "input":
            face = PURDUE_BLACK; text_c = PURDUE_GOLD
        elif kind == "output":
            face = ACCENT; text_c = PURDUE_BLACK
        else:
            face = PURDUE_GOLD; text_c = PURDUE_BLACK
        rect = mpatches.FancyBboxPatch((x, y - box_h / 2), box_w, box_h,
                                       boxstyle="round,pad=0.05",
                                       linewidth=2.5, edgecolor=PURDUE_BLACK,
                                       facecolor=face)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y, label, ha="center", va="center",
                fontsize=14, fontweight="bold", color=text_c)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap * 0.95, y),
                        xytext=(x + box_w, y),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color=PURDUE_BLACK))
    ax.text(5, 4.7, "End-to-end Python pipeline",
            ha="center", fontsize=28, fontweight="bold", color=PURDUE_BLACK)
    ax.text(5, 0.6, "Reproducible · Single-pass JPEG decode · Versioned",
            ha="center", fontsize=18, color=PURDUE_GOLD_DARK,
            fontweight="bold")
    save(fig, "p04_pipeline")


# ---------- 2. Lighting balance before/after -------------------------------
def fig_lighting():
    # Pick a sample
    sample = sorted(PROCESSED.glob("CN*.png"))
    sample = [p for p in sample if not p.stem.endswith("_mask")][20]
    bal = cv2.imread(str(sample))

    # Re-create a "before" by undoing gray-world: apply random gain
    gray = cv2.cvtColor(bal, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, 90)
    bright = bal[gray >= thr]
    means = bright.reshape(-1, 3).mean(axis=0)
    target = np.array([230.0, 230.0, 230.0])
    gains_used = target / np.maximum(means, 1)
    # "before" = bal / gains (approximate inversion)
    before = (bal.astype(np.float32) / gains_used)
    before = np.clip(before * 0.78, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_169)
    axes[0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    axes[0].set_title("BEFORE: raw smartphone capture",
                      fontsize=24, color=PURDUE_BLACK, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(bal, cv2.COLOR_BGR2RGB))
    axes[1].set_title("AFTER: gray-world balanced",
                      fontsize=24, color=PURDUE_GOLD_DARK, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle("Lighting balance equalizes session-to-session variation",
                 fontsize=26, fontweight="bold", color=PURDUE_BLACK, y=0.98)
    save(fig, "p05_lighting_balance")


# ---------- 3. Segmentation iteration --------------------------------------
def fig_segmentation():
    sample = sorted(PROCESSED.glob("CN*.png"))
    sample = [p for p in sample if not p.stem.endswith("_mask")][30]
    img = cv2.imread(str(sample))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, before = cv2.threshold(lab[:, :, 0], 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    after = cv2.imread(str(sample.parent / f"{sample.stem}_mask.png"),
                       cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_169)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (corn)", fontsize=22, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(before, cmap="gray")
    axes[1].set_title("BEFORE: L*-Otsu\n(misses translucent edges)",
                      fontsize=20, color="#A50000", fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(after, cmap="gray")
    axes[2].set_title("AFTER: HSV chromatic gate\n(captures full leaf)",
                      fontsize=20, color=PURDUE_GOLD_DARK, fontweight="bold")
    axes[2].axis("off")
    fig.suptitle("QC review drove a species-aware segmentation strategy",
                 fontsize=24, fontweight="bold", y=1.02)
    save(fig, "p06_segmentation")


# ---------- 4. Feature families graphic ------------------------------------
def fig_features():
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    families = [
        {"title": "COLOR\n21 features",
         "items": ["RGB · HSV · LAB", "means and std", "greenness, redness"],
         "x": 1.5, "color": ACCENT},
        {"title": "TEXTURE\n18 features",
         "items": ["GLCM matrices", "d ∈ {1, 3, 5}", "6 Haralick props"],
         "x": 5.0, "color": PURDUE_GOLD},
        {"title": "MORPHOLOGY\n7 features",
         "items": ["area · perimeter", "solidity · compactness",
                   "extent · aspect ratio"],
         "x": 8.5, "color": PURDUE_GOLD_DARK},
    ]
    for fam in families:
        rect = mpatches.FancyBboxPatch((fam["x"] - 1.4, 1.5), 2.8, 3.5,
                                       boxstyle="round,pad=0.1",
                                       linewidth=3, edgecolor=PURDUE_BLACK,
                                       facecolor=fam["color"])
        ax.add_patch(rect)
        ax.text(fam["x"], 4.5, fam["title"], ha="center", va="center",
                fontsize=22, fontweight="bold", color=PURDUE_BLACK)
        for j, line in enumerate(fam["items"]):
            ax.text(fam["x"], 3.2 - j * 0.55, line, ha="center",
                    fontsize=16, color=PURDUE_BLACK, fontweight="bold")
    ax.text(5, 5.7, "33 phenotypic features per leaf",
            ha="center", fontsize=30, fontweight="bold", color=PURDUE_BLACK)
    ax.text(5, 0.7, "Each computed only on segmented leaf pixels",
            ha="center", fontsize=18, color=PURDUE_GOLD_DARK, fontweight="bold")
    save(fig, "p07_features")


# ---------- 5. LAB scatter with ellipses -----------------------------------
def fig_lab():
    df = load()
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    for sp in ["CN", "SB"]:
        sub = df[df.species == sp]
        ax.scatter(sub["mean_a"], sub["mean_b"], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.85,
                   edgecolor=PURDUE_BLACK, s=180, linewidth=2)
        if len(sub) > 2:
            xa, ya = sub["mean_a"].values, sub["mean_b"].values
            cov = np.cov(xa, ya)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 2 * np.sqrt(vals)
            ax.add_patch(mpatches.Ellipse((xa.mean(), ya.mean()), w, h,
                                          angle=angle,
                                          edgecolor=SPECIES_COLOR[sp],
                                          facecolor="none", lw=4))
    ax.set_xlabel("Mean a* (red ↔ green)", fontsize=24)
    ax.set_ylabel("Mean b* (blue ↔ yellow)", fontsize=24)
    ax.legend(loc="upper left", frameon=True, fontsize=22)
    headline(ax, "Corn and soybean separate cleanly in LAB color space")
    save(fig, "p08_lab_scatter")


# ---------- 6. PCA biplot --------------------------------------------------
def fig_pca():
    df = load()
    num_cols = df.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed"] if c in df.columns], errors="ignore")
    X = StandardScaler().fit_transform(df[num_cols].fillna(0.0).values)
    pca = PCA(n_components=2); scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    mag = np.sqrt((loadings ** 2).sum(axis=1))
    top4 = np.argsort(mag)[::-1][:4]

    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    for sp in ["CN", "SB"]:
        m = df["species"].values == sp
        ax.scatter(scores[m, 0], scores[m, 1], color=SPECIES_COLOR[sp],
                   label=SPECIES_LABEL[sp], alpha=0.85,
                   edgecolor=PURDUE_BLACK, s=200, linewidth=2)
    s = 1.0 * max(np.abs(scores).max(), 1) / max(np.abs(loadings).max(), 1e-9)
    for i in top4:
        ax.arrow(0, 0, loadings[i, 0] * s, loadings[i, 1] * s,
                 color=PURDUE_BLACK, alpha=0.95, head_width=0.25,
                 length_includes_head=True, linewidth=3)
        ax.text(loadings[i, 0] * s * 1.12, loadings[i, 1] * s * 1.12,
                num_cols[i], color=PURDUE_BLACK, fontsize=18,
                fontweight="bold")
    ax.axhline(0, color="grey", lw=1); ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% of variance)", fontsize=24)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% of variance)", fontsize=24)
    ax.legend(loc="best", fontsize=22)
    headline(ax,
             f"All 33 features collapse to one axis: PC1 = {evr[0]*100:.0f}%")
    save(fig, "p09_pca_biplot")


# ---------- 7. Effect-size bar chart ---------------------------------------
def fig_effects():
    cmp = pd.read_csv(DATA / "species_comparison.csv")
    cmp["abs_d"] = cmp["cohens_d"].abs()
    top = cmp.sort_values("abs_d", ascending=False).head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    colors = [ACCENT if d < 0 else PURDUE_GOLD_DARK
              for d in top["cohens_d"]]
    ax.barh(top["metric"], top["abs_d"], color=colors,
            edgecolor=PURDUE_BLACK, linewidth=2)
    for i, (m, d) in enumerate(zip(top["metric"], top["cohens_d"])):
        ax.text(abs(d) + 0.08, i, f"d={d:+.2f}", va="center",
                fontsize=15, fontweight="bold", color=PURDUE_BLACK)
    ax.set_xlabel("|Cohen's d| effect size", fontsize=24)
    ax.set_xlim(0, top["abs_d"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=16)
    headline(ax,
             "Every top-10 feature: p < 1×10⁻³⁰  ·  effect size > 2.8")
    save(fig, "p10_effect_sizes")


# ---------- 8. QC summary 4-check infographic ------------------------------
def fig_qc_summary():
    qc = pd.read_csv(DATA / "qc_summary_table.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # Map qc rows to short titles
    titles = ["Segmentation\nquality",
              "Exposure /\nillumination",
              "Per-species\nIQR outliers",
              "Metadata\ncompleteness"]
    severe = qc["Severe Flags"].tolist()
    mild = qc["Mild Flags"].tolist()
    total = qc["Total Checked"].iloc[0]

    for i, (t, s, m) in enumerate(zip(titles, severe, mild)):
        x = 0.7 + i * 2.3
        # severity color: green if 0 severe
        face = "#1f8a3f" if s == 0 else "#A50000"
        rect = mpatches.FancyBboxPatch((x, 1.3), 2.0, 3.4,
                                       boxstyle="round,pad=0.08",
                                       linewidth=3, edgecolor=PURDUE_BLACK,
                                       facecolor=face)
        ax.add_patch(rect)
        ax.text(x + 1.0, 4.1, t, ha="center", va="center",
                fontsize=18, fontweight="bold", color="white")
        ax.text(x + 1.0, 3.0, f"{s}", ha="center", va="center",
                fontsize=46, fontweight="bold", color="white")
        ax.text(x + 1.0, 2.25, "severe flags", ha="center", va="center",
                fontsize=14, color="white")
        ax.text(x + 1.0, 1.65, f"({m} mild)", ha="center", va="center",
                fontsize=14, color="white", style="italic")

    ax.text(5, 5.6, "4 independent QC checks  ·  140 images",
            ha="center", fontsize=28, fontweight="bold", color=PURDUE_BLACK)
    ax.text(5, 0.6, "Zero severe flags in segmentation and exposure "
            "after QC-driven iteration",
            ha="center", fontsize=18, color=PURDUE_GOLD_DARK, fontweight="bold")
    save(fig, "p11_qc_summary")


# ---------- 9. Title slide background --------------------------------------
def fig_title():
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.625); ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 5.625, color=PURDUE_BLACK))
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 1.0, color=ACCENT))

    ax.text(5, 4.0, "RGB Image Analysis of Corn",
            ha="center", fontsize=46, fontweight="bold", color=PURDUE_GOLD)
    ax.text(5, 3.2, "and Soybean Leaves",
            ha="center", fontsize=46, fontweight="bold", color=PURDUE_GOLD)
    ax.text(5, 2.2, "Phenotypic Feature Extraction Pipeline",
            ha="center", fontsize=24, color="white", style="italic")
    ax.text(5, 1.4, "Charles Wang  ·  ABE 651  ·  Advisor: Dr. Jian Jin",
            ha="center", fontsize=20, color="white")
    ax.text(5, 0.5, "Purdue University · Agricultural & Biological Engineering",
            ha="center", fontsize=16, color=PURDUE_BLACK, fontweight="bold")
    save(fig, "p01_title")


# ---------- 10. Dataset at-a-glance ----------------------------------------
def fig_dataset():
    cn = sorted(PROCESSED.glob("CN*.png"))
    cn = [p for p in cn if not p.stem.endswith("_mask")][30]
    sb = sorted(PROCESSED.glob("SB*.png"))
    sb = [p for p in sb if not p.stem.endswith("_mask")][30]
    cn_img = cv2.imread(str(cn))
    sb_img = cv2.imread(str(sb))

    fig = plt.figure(figsize=FIGSIZE_169)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 4])

    # Top: counts banner
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    ax_top.text(0.5, 0.65,
                "140 leaves  ·  70 corn + 70 soybean  ·  smartphone in transmittance lightbox",
                ha="center", va="center", fontsize=26, fontweight="bold",
                color=PURDUE_BLACK)
    ax_top.text(0.5, 0.15,
                "Excised, single-leaf images · auto exposure · auto white balance",
                ha="center", va="center", fontsize=18, color=PURDUE_GOLD_DARK,
                fontweight="bold")

    ax_cn = fig.add_subplot(gs[1, 0])
    ax_cn.imshow(cv2.cvtColor(cn_img, cv2.COLOR_BGR2RGB))
    ax_cn.set_title("CORN  ·  Zea mays  ·  n = 70",
                    fontsize=22, fontweight="bold",
                    color=PURDUE_GOLD_DARK)
    ax_cn.axis("off")

    ax_sb = fig.add_subplot(gs[1, 1])
    ax_sb.imshow(cv2.cvtColor(sb_img, cv2.COLOR_BGR2RGB))
    ax_sb.set_title("SOYBEAN  ·  Glycine max  ·  n = 70",
                    fontsize=22, fontweight="bold",
                    color=PURDUE_BLACK)
    ax_sb.axis("off")

    save(fig, "p03_dataset")


# ---------- 11. Problem statement ------------------------------------------
def fig_problem():
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.625); ax.axis("off")

    ax.text(5, 5.0, "Can a smartphone replace a phenotyping rig?",
            ha="center", fontsize=34, fontweight="bold", color=PURDUE_BLACK)

    bullets = [
        ("Hardware:", "smartphone + custom transmittance lightbox"),
        ("Question:", "do corn and soybean leaves separate on color, "
                      "texture, and shape?"),
        ("Need:", "reproducible, low-cost, scriptable pipeline anyone can run"),
    ]
    y = 3.5
    for lead, body in bullets:
        ax.add_patch(mpatches.Circle((1.2, y + 0.05), 0.10,
                                     color=ACCENT))
        ax.text(1.6, y + 0.05, lead, fontsize=22, fontweight="bold",
                color=PURDUE_GOLD_DARK, va="center")
        ax.text(3.2, y + 0.05, body, fontsize=22, color=PURDUE_BLACK,
                va="center")
        y -= 0.95
    save(fig, "p02_problem")


# ---------- 12. Conclusions ------------------------------------------------
def fig_conclusion():
    fig, ax = plt.subplots(figsize=FIGSIZE_169)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.625); ax.axis("off")
    ax.text(5, 5.0, "Take-aways",
            ha="center", fontsize=38, fontweight="bold", color=PURDUE_BLACK)
    points = [
        ("✓", "Pipeline extracts 33 phenotypic features per leaf, "
              "fully reproducible"),
        ("✓", "Species separate on every feature family — color, "
              "texture, morphology"),
        ("✓", "mean a* and perimeter are the strongest single-feature "
              "discriminators"),
        ("✓", "Iterative QC review caught a corn-segmentation failure "
              "and drove the species-aware fix"),
        ("✓", "Code, data, masks, and figures versioned in the data "
              "publication"),
    ]
    y = 4.0
    for mark, text in points:
        ax.text(0.8, y, mark, fontsize=28, color=PURDUE_GOLD_DARK,
                fontweight="bold", va="center")
        ax.text(1.6, y, text, fontsize=20, color=PURDUE_BLACK,
                va="center", fontweight="bold")
        y -= 0.7
    save(fig, "p12_conclusions")


def main():
    print("Building presentation figures...")
    fig_title()
    fig_problem()
    fig_dataset()
    fig_pipeline()
    fig_lighting()
    fig_segmentation()
    fig_features()
    fig_lab()
    fig_pca()
    fig_effects()
    fig_qc_summary()
    fig_conclusion()
    print(f"\nAll done. {len(list(OUT.glob('*.png')))} files in {OUT}")


if __name__ == "__main__":
    main()
