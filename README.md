# Leaf RGB Phenotypic Feature Extraction

**ABE 651 — Final Project | Charles Wang | Department of Agricultural & Biological Engineering, Purdue University | Advisor: Dr. Jian Jin**

This repository extracts a curated set of 20 phenotypic features (colour, GLCM texture, morphology) from RGB smartphone photographs of corn (*Zea mays*) and soybean (*Glycine max*) leaves captured inside a custom transmittance lightbox. The deliverable is a **reusable phenotypic feature dataset** — not a comparison of the two species. Species (CN, SB) is a labeling/grouping variable only.

The full narrative is in [docs/ABE651_Final_Report.docx](docs/ABE651_Final_Report.docx); the dataset itself is published on PURR (link to be inserted on final publication).

---

## Quickstart

```bash
git clone <repo-url> "Final Report"
cd "Final Report"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place raw JPEGs in raw/Corn Images/ and raw/Soybean Images/ then:
python scripts/00_rename.py            # standardize filenames
python scripts/01_inventory.py         # inventory + EXIF metadata
python scripts/02_segment.py           # downsample, gray-world, segment
python scripts/03_extract_features.py  # 20 features per leaf
python scripts/04_eda.py               # 7 exploratory figures
python scripts/05_quality_check.py     # QC flags + figures
python scripts/06_summary_stats.py     # per-species descriptive table
```

Each step writes its outputs to disk and can be re-run independently. Intermediate state lives in `processed/`, `data/`, and `figures/` — never overwriting `raw/`.

---

## Repository layout

```
Final Report/
├── raw/                    # original JPEGs — READ-ONLY
│   ├── Corn Images/        # 70 corn leaves (CN001..CN070)
│   └── Soybean Images/     # 70 soybean leaves (SB001..SB070)
├── processed/              # PNG working images + binary masks
├── data/                   # extracted CSVs (see Outputs below)
├── figures/                # EDA + QC + publication PNG/SVG
├── scripts/                # the pipeline (00–09 + helpers)
├── docs/                   # final report .docx + logs
├── requirements.txt        # pinned Python dependencies
├── CLAUDE.md               # internal pipeline spec / contributor guide
└── README.md               # this file
```

---

## Pipeline scripts

| Step | Script | Purpose |
|---|---|---|
| 00 | [scripts/00_rename.py](scripts/00_rename.py) | Rename and standardize raw JPEGs to `[SampleID]_[Species]_[Date]_[Seq].jpg` |
| 01 | [scripts/01_inventory.py](scripts/01_inventory.py) | Walk `raw/`, parse filenames + EXIF, write `metadata_inventory.csv` |
| 02 | [scripts/02_segment.py](scripts/02_segment.py) | Downsample, gray-world balance, segment leaf, write masks + QC |
| 03 | [scripts/03_extract_features.py](scripts/03_extract_features.py) | Extract colour, GLCM texture, and morphology features per leaf |
| 04 | [scripts/04_eda.py](scripts/04_eda.py) | Generate seven exploratory figures (`fig01–fig07`) |
| 05 | [scripts/05_quality_check.py](scripts/05_quality_check.py) | Apply QC checks, flag outliers, produce QC figures and summary |
| 06 | [scripts/06_summary_stats.py](scripts/06_summary_stats.py) | Per-species descriptive statistics (primary) plus a supplementary 2-sample appendix |

---

## Curated feature set (20 features per leaf)

| Block | Features |
|---|---|
| RGB (5) | `mean_R`, `mean_G`, `mean_B`, `greenness_idx = G/(R+G+B)`, `redness_idx = R/(R+G+B)` |
| LAB (6) | `mean_L`, `mean_a`, `mean_b`, `std_L`, `std_a`, `std_b` |
| GLCM @ d=1 (4) | `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation` (averaged over 4 angles) |
| Morphology (5) | `area_px`, `aspect_ratio`, `extent`, `solidity`, `compactness` |

Features removed during curation (HSV block, RGB std, multi-distance GLCM, dissimilarity, ASM, perimeter, equivalent diameter) are documented in Section 5.1 of the final report. Do not re-add them without rationale.

---

## Outputs

**Data files** (`data/`):
- `metadata_inventory.csv` — per-image inventory + EXIF
- `leaf_metrics.csv` — 20 features × 140 leaves (the primary feature table)
- `segmentation_qc.csv` — per-image segmentation diagnostics
- `summary_statistics.csv` — **the headline output**: per-species N, mean, SD, median, Q1, Q3, min, max, CV for every feature
- `qc_summary_table.csv` — flag counts per QC check
- `species_comparison_supplementary.csv` — Welch's t and Mann-Whitney U descriptors (appendix only, not interpreted)
- `failed_images.txt` — list of images that could not be opened (may be empty)

**Figures** (`figures/`): seven EDA figures (`fig01–fig07`), three QC figures (`qc_fig01–qc_fig03`), and publication-quality re-renders (`report_*.png/.svg`).

**Documents** (`docs/`): `ABE651_Final_Report.docx`, `Final_Data_Summary.docx`, `pipeline_log.txt`.

---

## Reproducibility

- Every script logs start/end and a one-line summary to `docs/pipeline_log.txt` (append-mode, timestamped).
- `tqdm` progress bars on every loop over images.
- `requirements.txt` pins major versions; tested with Python 3.11 on macOS.
- `raw/` is treated as read-only — re-running any step never overwrites raw inputs.
- The companion file `interim_snapshot/` (if present) contains archived outputs from an earlier project iteration; leave it alone.

---

## Data publication

The complete dataset (raw images, processed PNGs, masks, all CSVs, figures, and a code archive) is published on the **Purdue University Research Repository (PURR)** at `purr.purdue.edu` (DOI to be assigned on final publication). The PURR layout mirrors this repository so a downloader can re-run the pipeline without modification.

License for the published data: **CC BY 4.0** (attribution).

---

## Citation

If you use this dataset or pipeline, please cite:

> Wang, C. (2026). *RGB Image Analysis of Corn and Soybean Leaves for Phenotypic Feature Extraction.* Purdue University Research Repository. \[DOI to be assigned\]

---

## Contact

Charles Wang — abepurdue107@gmail.com
Department of Agricultural & Biological Engineering, Purdue University
