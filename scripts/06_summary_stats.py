"""Step 6 — per-species descriptive statistics (primary) plus a supplementary
two-sample comparison table (appendix only).

The project goal is feature extraction and dataset characterization, not species
comparison. The headline output is the per-species descriptive table. Inferential
tests are computed for completeness only and saved to a supplementary CSV that is
not featured in the report narrative or figures.
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import DATA, log


def main() -> None:
    log("06_summary_stats", "start")
    df = pd.read_csv(DATA / "leaf_metrics.csv")
    df = df[df["seg_failed"] == False].copy()
    numeric = df.select_dtypes(include=np.number).columns.drop(
        [c for c in ["seg_failed"] if c in df.columns], errors="ignore")

    # --- Primary output: per-species descriptive statistics ---
    rows = []
    for sp, sub in df.groupby("species"):
        for col in numeric:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            cv = float(std / mean) if mean != 0 else np.nan
            rows.append({
                "species": sp, "metric": col, "N": len(vals),
                "mean": mean, "std": std,
                "median": float(np.median(vals)),
                "Q1": float(np.percentile(vals, 25)),
                "Q3": float(np.percentile(vals, 75)),
                "min": float(vals.min()), "max": float(vals.max()),
                "cv": cv,
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(DATA / "summary_statistics.csv", index=False)

    # --- Supplementary appendix: basic two-sample descriptors ---
    # Not featured in figures or narrative. No ANOVA, no effect-size emphasis.
    comp_rows = []
    cn = df[df["species"] == "CN"]
    sb = df[df["species"] == "SB"]
    for col in numeric:
        a = cn[col].dropna().values
        b = sb[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        comp_rows.append({
            "metric": col,
            "welch_t": float(t), "welch_p": float(pt),
            "mwu_U": float(u), "mwu_p": float(pu),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(DATA / "species_comparison_supplementary.csv", index=False)

    # Remove the legacy file if it exists, since the project no longer features it
    legacy = DATA / "species_comparison.csv"
    if legacy.exists():
        legacy.unlink()

    print("\n=== Per-species descriptive statistics (head) ===")
    print(summary_df.head(20).to_string(index=False))
    print(f"\nTotal descriptive rows: {len(summary_df)}")
    print(f"Supplementary comparison rows (appendix only): {len(comp_df)}")
    log("06_summary_stats",
        f"descriptive rows={len(summary_df)}, supplementary comparisons={len(comp_df)}")


if __name__ == "__main__":
    main()
