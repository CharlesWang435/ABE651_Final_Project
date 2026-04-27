"""Step 0 — rename & standardize raw images.

Copies raw JPEGs from raw/Corn Images and raw/Soybean Images into raw/
with the convention [SampleID]_[Species]_[YYYYMMDD]_[SeqNum].jpg and writes
a rename log.
"""
import re
import shutil
from pathlib import Path
from PIL import Image, ExifTags
from common import RAW, DOCS, log

SPECIES_FOLDERS = {"Corn Images": "CN", "Soybean Images": "SB"}
EXIF_DATE_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal")


def extract_date(path: Path) -> str:
    m = re.match(r"(\d{8})", path.name)
    if m:
        return m.group(1)
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        raw = exif.get(EXIF_DATE_TAG)
        if raw:
            return raw.split(" ")[0].replace(":", "")
    except Exception:
        pass
    return "00000000"


def main() -> None:
    log("00_rename", "start")
    log_lines = []
    seq = 0
    species_seq = {"CN": 0, "SB": 0}
    for folder_name, sp in SPECIES_FOLDERS.items():
        src_dir = RAW / folder_name
        if not src_dir.exists():
            log("00_rename", f"missing folder {src_dir}")
            continue
        files = sorted([p for p in src_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg")])
        for p in files:
            species_seq[sp] += 1
            seq += 1
            sample_id = f"{sp}{species_seq[sp]:03d}"
            date_str = extract_date(p)
            new_name = f"{sample_id}_{sp}_{date_str}_{seq:04d}.jpg"
            dst = RAW / new_name
            if not dst.exists():
                shutil.copy2(p, dst)
            log_lines.append(f"{p.relative_to(RAW)} -> {new_name}")
    (DOCS / "rename_log.txt").write_text("\n".join(log_lines) + "\n")
    log("00_rename", f"renamed {len(log_lines)} files "
                   f"(CN={species_seq['CN']}, SB={species_seq['SB']})")


if __name__ == "__main__":
    main()
