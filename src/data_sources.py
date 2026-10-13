"""Load the frozen attributed snapshot after byte-level integrity verification."""

import hashlib
import io
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_official():
    raw = (ROOT / "data/official_fx.csv").read_bytes()
    manifest = json.loads(
        (ROOT / "data/official_fx_manifest.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(raw).hexdigest() != manifest["file_sha256"]:
        raise ValueError("Official snapshot checksum mismatch")
    frame = pd.read_csv(io.BytesIO(raw))
    frame.attrs["source_manifest"] = manifest
    return frame
