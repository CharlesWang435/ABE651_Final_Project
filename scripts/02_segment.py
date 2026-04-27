"""Step 2 — preprocessing, gray-world lighting balance, segmentation."""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from common import RAW, PROCESSED, DATA, SEG_PREVIEWS, log

TARGET_LONG_EDGE = 2000
TARGET_NEUTRAL = np.array([230.0, 230.0, 230.0])   # B, G, R


def downsample(img):
    h, w = img.shape[:2]
    s = TARGET_LONG_EDGE / max(h, w)
    if s < 1.0:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def gray_world_balance(img):
    """Rescale so the top 10% brightest pixels have per-channel mean = TARGET_NEUTRAL."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, 90)
    bright = img[gray >= thr]
    if bright.size == 0:
        return img
    means = bright.reshape(-1, 3).mean(axis=0)
    means[means < 1] = 1
    gains = TARGET_NEUTRAL / means
    balanced = img.astype(np.float32) * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _clean_mask(raw_mask, close_kernel=11, min_size=2000):
    """Open (remove speckle) → close → keep largest component → fill holes → smooth."""
    # Opening removes isolated speckle in the background before closing bridges it
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m_bool = remove_small_objects(m.astype(bool), min_size=min_size)
    lbl = label(m_bool)
    if lbl.max() == 0:
        return np.zeros_like(raw_mask)
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    keep = (lbl == sizes.argmax())
    keep = binary_fill_holes(keep)
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    keep_u8 = cv2.morphologyEx(keep.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_smooth)
    keep_u8 = binary_fill_holes(keep_u8 > 0).astype(np.uint8) * 255
    return keep_u8


def _segment_corn(img):
    """Corn: HSV chromatic-green gate AND positive Excess-Green.

    Corn leaves are thin at the edge and transmit bright light, which fools
    an L*-Otsu threshold. A pure-greenness ratio also fails because it admits
    slightly-tinted white paper and cannot reject the black lightbox frame.

    The fix is a three-part HSV gate that keeps only pixels that are
    genuinely chromatic green:
      - Hue in [25, 95]      → yellow-green to green band (OpenCV H=0..180)
      - Saturation >= 30     → rejects neutral white paper and grey backdrop
      - Value >= 25          → rejects the near-black lightbox frame
    AND then require ExG = 2G - R - B > 0 so the pixel is positively
    green-biased in linear RGB. The intersection gives a clean leaf
    mask on every imaging session.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Chlorophyll green has S roughly 80-220. White paper + JPEG noise
    # rarely exceeds S=50. 70 is a safe margin that keeps even thin,
    # translucent leaf edges while rejecting slightly-tinted white paper.
    m_hsv = ((H >= 25) & (H <= 95) & (S >= 70) & (V >= 25))

    b, g, r = cv2.split(img.astype(np.float32))
    m_exg = (2.0 * g - r - b) > 0.0

    raw = (m_hsv & m_exg).astype(np.uint8) * 255
    return _clean_mask(raw)


def _segment_soybean(img):
    """Soybean: L*-Otsu (dense leaf, well-behaved against bright backdrop)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    _, m = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _clean_mask(m)


def segment_leaf(img, species):
    if species == "CN":
        return _segment_corn(img)
    return _segment_soybean(img)


def qc_metrics(mask):
    h, w = mask.shape
    area = int((mask > 0).sum())
    total = h * w
    leaf_frac = area / total if total else 0.0
    solidity = 0.0
    bb_fill = 0.0
    if area > 0:
        lbl = label(mask > 0)
        rp = regionprops(lbl)[0]
        solidity = float(rp.solidity)
        minr, minc, maxr, maxc = rp.bbox
        bb_area = max((maxr - minr) * (maxc - minc), 1)
        bb_fill = area / bb_area
    return leaf_frac, solidity, bb_fill


def make_preview(orig, balanced, mask, out_path):
    overlay = balanced.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([60, 200, 60])).astype(np.uint8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)); ax[1].set_title("Balanced"); ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); ax[2].set_title("Mask overlay"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log("02_segment", "start")
    files = sorted([p for p in RAW.iterdir()
                    if p.is_file() and p.suffix.lower() == ".jpg"])
    rows = []
    failed = []
    for idx, p in enumerate(tqdm(files, desc="segment")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(p.name)
            continue
        ds = downsample(img)
        bal = gray_world_balance(ds)
        cv2.imwrite(str(PROCESSED / f"{p.stem}.png"), bal)
        species = p.name.split("_")[1].upper() if "_" in p.name else "SB"
        mask = segment_leaf(bal, species)
        cv2.imwrite(str(PROCESSED / f"{p.stem}_mask.png"), mask)
        lf, sol, bbf = qc_metrics(mask)
        rows.append({"filename": p.name, "leaf_frac": lf,
                     "solidity": sol, "bbox_fill": bbf,
                     "seg_failed": bool((mask > 0).sum() == 0)})
        if idx % 10 == 0:
            make_preview(ds, bal, mask, SEG_PREVIEWS / f"{p.stem}_preview.png")
    pd.DataFrame(rows).to_csv(DATA / "segmentation_qc.csv", index=False)
    (DATA / "failed_images.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    log("02_segment", f"segmented {len(rows)} images; failed={len(failed)}")


if __name__ == "__main__":
    main()
