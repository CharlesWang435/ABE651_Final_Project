"""Shared helpers: logging, paths, colour palette."""
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
SEG_PREVIEWS = FIGURES / "segmentation_previews"

for d in (PROCESSED, DATA, FIGURES, DOCS, SEG_PREVIEWS):
    d.mkdir(parents=True, exist_ok=True)

SPECIES_COLOR = {"SB": "#2d8a4e", "CN": "#e8a020"}
SPECIES_LABEL = {"SB": "Soybean", "CN": "Corn"}

PIPELINE_LOG = DOCS / "pipeline_log.txt"


def log(script_name: str, message: str) -> None:
    """Append a timestamped log line to docs/pipeline_log.txt."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PIPELINE_LOG, "a") as f:
        f.write(f"[{ts}] {script_name}: {message}\n")
    print(f"[{ts}] {script_name}: {message}")
