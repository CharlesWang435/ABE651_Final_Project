"""Step 1 — inventory and EXIF metadata for all standardized images."""
import re
import pandas as pd
from PIL import Image, ExifTags
from common import RAW, DATA, log

EXIF_MAP = {
    "iso": "ISOSpeedRatings",
    "exposure_time": "ExposureTime",
    "white_balance": "WhiteBalance",
    "gps": "GPSInfo",
}
TAG_ID = {v: k for k, v in ExifTags.TAGS.items()}
STD = re.compile(r"^(?P<sid>[A-Z]{2}\d{3})_(?P<sp>CN|SB)_(?P<date>\d{8})_(?P<seq>\d{3,4})\.jpg$",
                 re.IGNORECASE)


def main() -> None:
    log("01_inventory", "start")
    rows = []
    for p in sorted(RAW.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        m = STD.match(p.name)
        if not m:
            continue
        row = {"filename": p.name,
               "sample_id": m.group("sid"),
               "species": m.group("sp").upper(),
               "collection_date": m.group("date"),
               "seq_num": int(m.group("seq")),
               "iso": None, "exposure_time": None, "white_balance": None,
               "file_size_mb": round(p.stat().st_size / 1_048_576, 3),
               "image_width_px": None, "image_height_px": None,
               "exif_notes": ""}
        try:
            img = Image.open(p)
            row["image_width_px"], row["image_height_px"] = img.size
            exif = img._getexif() or {}
            for col, name in EXIF_MAP.items():
                tid = TAG_ID.get(name)
                if tid and tid in exif:
                    val = exif[tid]
                    if col == "exposure_time" and hasattr(val, "__float__"):
                        val = float(val)
                    if col == "gps":
                        row["exif_notes"] += "gps_present;"
                    else:
                        row[col] = val
        except Exception as e:
            row["exif_notes"] += f"err:{type(e).__name__};"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "metadata_inventory.csv", index=False)
    log("01_inventory", f"inventoried {len(df)} images | "
                       f"per-species={df['species'].value_counts().to_dict()} | "
                       f"date range {df['collection_date'].min()}–{df['collection_date'].max()}")


if __name__ == "__main__":
    main()
