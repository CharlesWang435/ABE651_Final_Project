"""Step 3 — colour, texture, and morphology feature extraction.

Feature set is deliberately curated. Redundant or non-interpretable features have been
removed (HSV block, RGB std, multi-distance GLCM, dissimilarity, ASM, perimeter,
equivalent_diameter) — see the report for the rationale.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label, perimeter as sk_perimeter
from common import RAW, PROCESSED, DATA, log

GLCM_DISTANCE = 1
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def color_stats(img_bgr, mask_bool):
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    b_m, g_m, r_m = b[mask_bool], g[mask_bool], r[mask_bool]
    total = r_m.astype(np.float64) + g_m + b_m + 1e-9
    out = {
        "mean_R": float(r_m.mean()), "mean_G": float(g_m.mean()), "mean_B": float(b_m.mean()),
        "greenness_idx": float((g_m / total).mean()),
        "redness_idx": float((r_m / total).mean()),
    }
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # OpenCV LAB: L in 0-255 -> rescale to CIE 0-100; a,b 0-255 -> -128..127
    L = lab[:, :, 0][mask_bool].astype(np.float32) * (100.0 / 255.0)
    a = lab[:, :, 1][mask_bool].astype(np.float32) - 128.0
    bb = lab[:, :, 2][mask_bool].astype(np.float32) - 128.0
    out.update({"mean_L": float(L.mean()), "mean_a": float(a.mean()), "mean_b": float(bb.mean()),
                "std_L": float(L.std()), "std_a": float(a.std()), "std_b": float(bb.std())})
    return out


def texture_stats(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray[mask == 0] = 0
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {f"glcm_{p}": 0.0 for p in GLCM_PROPS}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray_crop = gray[y0:y1, x0:x1]
    gray_q = (gray_crop // 8).astype(np.uint8)  # 32 gray levels
    glcm = graycomatrix(gray_q, distances=[GLCM_DISTANCE], angles=GLCM_ANGLES,
                        levels=32, symmetric=True, normed=True)
    out = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)  # shape (1, n_angle)
        out[f"glcm_{prop}"] = float(vals.mean())
    return out


def morphology_stats(mask):
    m_bool = mask > 0
    area = int(m_bool.sum())
    if area == 0:
        return {"area_px": 0, "aspect_ratio": 0.0, "extent": 0.0,
                "solidity": 0.0, "compactness": 0.0}
    rp = regionprops(label(m_bool))[0]
    minr, minc, maxr, maxc = rp.bbox
    h = maxr - minr
    w = maxc - minc
    per = sk_perimeter(m_bool)
    return {"area_px": area,
            "aspect_ratio": float(h / max(w, 1)),
            "extent": float(rp.extent),
            "solidity": float(rp.solidity),
            "compactness": float(4 * np.pi * area / max(per ** 2, 1e-9))}


def main() -> None:
    log("03_extract_features", "start")
    inventory = pd.read_csv(DATA / "metadata_inventory.csv")
    rows = []
    for _, meta in tqdm(inventory.iterrows(), total=len(inventory), desc="features"):
        stem = Path(meta["filename"]).stem
        img = cv2.imread(str(PROCESSED / f"{stem}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(PROCESSED / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None or (mask > 0).sum() == 0:
            rows.append({"filename": meta["filename"], "sample_id": meta["sample_id"],
                         "species": meta["species"], "seg_failed": True})
            continue
        mb = mask > 0
        row = {"filename": meta["filename"], "sample_id": meta["sample_id"],
               "species": meta["species"], "seg_failed": False}
        row.update(color_stats(img, mb))
        row.update(texture_stats(img, mask))
        row.update(morphology_stats(mask))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "leaf_metrics.csv", index=False)
    log("03_extract_features", f"wrote {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
