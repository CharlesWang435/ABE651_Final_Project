"""Step 5 — data quality checks and QC figures."""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from common import DATA, FIGURES, PROCESSED, SPECIES_LABEL, log

sns.set_style("whitegrid")


def outlier_flags(values, q1=0.25, q3=0.75):
    a = np.asarray(values, dtype=float)
    Q1 = np.nanpercentile(a, 25)
    Q3 = np.nanpercentile(a, 75)
    IQR = Q3 - Q1
    mild_lo, mild_hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    sev_lo, sev_hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
    mild = ((a < mild_lo) | (a > mild_hi)) & ((a >= sev_lo) & (a <= sev_hi))
    severe = (a < sev_lo) | (a > sev_hi)
    return mild, severe


def main() -> None:
    log("05_quality_check", "start")
    metrics = pd.read_csv(DATA / "leaf_metrics.csv")
    inventory = pd.read_csv(DATA / "metadata_inventory.csv")
    seg_qc = pd.read_csv(DATA / "segmentation_qc.csv")

    summary_rows = []

    # --- Check 1: segmentation quality ---
    m = metrics.copy()
    # leaf_frac: compute from segmentation_qc
    m = m.merge(seg_qc[["filename", "leaf_frac", "bbox_fill"]], on="filename", how="left")
    seg_flag = (m["leaf_frac"] < 0.05) | (m["leaf_frac"] > 0.9) | (m["solidity"] < 0.6)
    seg_severe = (m["leaf_frac"] < 0.02) | (m["leaf_frac"] > 0.95) | (m["solidity"] < 0.4)
    summary_rows.append({"QC Check": "Segmentation quality (area 5–90%, solidity>0.6)",
                         "Total Checked": len(m),
                         "Mild Flags": int((seg_flag & ~seg_severe).sum()),
                         "Severe Flags": int(seg_severe.sum()),
                         "Removed": 0,
                         "Notes": "Severe flags considered for removal"})

    # --- Check 2: exposure on CIE L* scale ---
    L = m["mean_L"].values
    exp_flag = (L > 85) | (L < 12)
    clipped = ((m[["mean_R", "mean_G", "mean_B"]] > 250).any(axis=1) |
               (m[["mean_R", "mean_G", "mean_B"]] < 5).any(axis=1))
    summary_rows.append({"QC Check": "Exposure / Illumination (L* 12-85, no clipped channels)",
                         "Total Checked": len(m),
                         "Mild Flags": int(clipped.sum()),
                         "Severe Flags": int(exp_flag.sum()),
                         "Removed": 0,
                         "Notes": "Clipped channels counted as mild"})

    # --- Check 3: per-species IQR outliers (union across numeric features) ---
    numeric = m.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed", "seq_num"] if c in m.columns], errors="ignore")
    mild_any = np.zeros(len(m), dtype=bool)
    sev_any = np.zeros(len(m), dtype=bool)
    for sp, sub in m.groupby("species"):
        idx = sub.index.values
        for col in numeric:
            mi, se = outlier_flags(sub[col].values)
            mild_any[idx[mi]] = True
            sev_any[idx[se]] = True
    # Don't double-count mild where severe
    mild_any = mild_any & ~sev_any
    summary_rows.append({"QC Check": "Metric outliers (per-species IQR)",
                         "Total Checked": len(m),
                         "Mild Flags": int(mild_any.sum()),
                         "Severe Flags": int(sev_any.sum()),
                         "Removed": 0,
                         "Notes": "Per-species Q1-1.5/3xIQR, Q3+1.5/3xIQR"})

    # --- Check 4: metadata completeness ---
    missing = int(inventory.isna().any(axis=1).sum())
    dup = int(inventory["sample_id"].duplicated().sum())
    summary_rows.append({"QC Check": "Metadata completeness & uniqueness",
                         "Total Checked": len(inventory),
                         "Mild Flags": missing,
                         "Severe Flags": dup,
                         "Removed": 0,
                         "Notes": "Rows with any NaN; duplicate sample_id"})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DATA / "qc_summary_table.csv", index=False)
    print(summary_df.to_string(index=False))

    # --- QC Figure 1: outlier scatter leaf_area vs greenness ---
    fig, ax = plt.subplots(figsize=(8, 6))
    color = np.where(sev_any, "red", np.where(mild_any, "orange", "steelblue"))
    ax.scatter(m["area_px"], m["greenness_idx"], c=color, alpha=0.75,
               edgecolor="black", s=40)
    for i in np.where(sev_any)[0]:
        ax.text(m["area_px"].iloc[i], m["greenness_idx"].iloc[i],
                m["filename"].iloc[i], fontsize=6, alpha=0.8)
    ax.set_xlabel("Leaf area (px²)")
    ax.set_ylabel("Greenness index")
    ax.set_title("QC Figure 1. Outlier detection (leaf area vs greenness)")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color="steelblue", label="normal"),
                       mp.Patch(color="orange", label="mild outlier"),
                       mp.Patch(color="red", label="extreme outlier")])
    plt.tight_layout()
    plt.savefig(FIGURES / "qc_fig01_outlier_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- QC Figure 2: exposure histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(L, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(12, color="red", linestyle="--")
    ax.axvline(85, color="red", linestyle="--")
    ymin, ymax = ax.get_ylim()
    ax.axvspan(0, 12, alpha=0.2, color="red")
    ax.axvspan(85, 100, alpha=0.2, color="red")
    ax.set_xlabel("Mean L* (CIE 0-100)")
    ax.set_ylabel("Image count")
    ax.set_title("QC Figure 2. Distribution of mean L* values")
    plt.tight_layout()
    plt.savefig(FIGURES / "qc_fig02_exposure_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- QC Figure 3: BEFORE/AFTER segmentation impact ---
    # "Before" = naive L*-Otsu mask on a corn image (the original pipeline)
    # "After"  = the QC-tuned HSV chromatic-gate mask (current pipeline)
    # This figure illustrates the *impact* of data-quality checking on the masks.
    corn_samples = m[m["species"] == "CN"].copy().head(3)
    fig, axes = plt.subplots(len(corn_samples), 3,
                             figsize=(13, 4 * max(len(corn_samples), 1)))
    if len(corn_samples) == 1:
        axes = np.array([axes])
    for i, (_, r) in enumerate(corn_samples.iterrows()):
        stem = Path(r["filename"]).stem
        orig = cv2.imread(str(PROCESSED / f"{stem}.png"))
        # BEFORE: naive L*-Otsu (the approach used in the v1 interim report)
        lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
        _, before = cv2.threshold(lab[:, :, 0], 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # AFTER: the saved tuned mask
        after = cv2.imread(str(PROCESSED / f"{stem}_mask.png"),
                           cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{r['filename']}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(before, cmap="gray")
        axes[i, 1].set_title("BEFORE QC: raw L*-Otsu", fontsize=10)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(after, cmap="gray")
        axes[i, 2].set_title("AFTER QC: HSV chromatic gate", fontsize=10)
        axes[i, 2].axis("off")
    plt.suptitle("QC Figure 3. Before/after impact of segmentation QC on corn masks",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "qc_fig03_before_after_segmentation.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- QC Figure 4: BEFORE/AFTER outlier removal on the metric distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    kept = m[~sev_any].copy()
    for ax, df_v, title in [(axes[0], m, f"BEFORE QC (n={len(m)})"),
                            (axes[1], kept, f"AFTER QC (n={len(kept)})")]:
        for sp, color in [("CN", "#e8a020"), ("SB", "#2d8a4e")]:
            sub = df_v[df_v["species"] == sp]
            ax.scatter([0 if sp == "CN" else 1] * len(sub) +
                       np.random.uniform(-0.18, 0.18, len(sub)),
                       sub["greenness_idx"], color=color, alpha=0.7,
                       edgecolor="black", s=35, label=sp)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Corn", "Soybean"])
        ax.set_title(title); ax.set_xlabel("Species")
    axes[0].set_ylabel("Greenness index G/(R+G+B)")
    plt.suptitle("QC Figure 4. Effect of extreme-outlier removal on greenness distribution",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES / "qc_fig04_outlier_impact.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Persist per-row flags for later use
    flags_df = m[["filename", "sample_id", "species"]].copy()
    flags_df["seg_flag"] = seg_flag.values
    flags_df["seg_severe"] = seg_severe.values
    flags_df["exp_flag"] = exp_flag
    flags_df["clipped_channel"] = clipped.values
    flags_df["outlier_mild"] = mild_any
    flags_df["outlier_severe"] = sev_any
    flags_df.to_csv(DATA / "qc_flags_per_image.csv", index=False)
    log("05_quality_check", f"summary rows={len(summary_df)}; "
                           f"severe_seg={int(seg_severe.sum())}; "
                           f"severe_exp={int(exp_flag.sum())}; "
                           f"severe_outliers={int(sev_any.sum())}")


if __name__ == "__main__":
    main()
