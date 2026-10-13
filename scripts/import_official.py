"""Download attributed H.10 reference rates and freeze a documented interval."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"EURUSD": "eu", "GBPUSD": "uk"}


def parse_rates(html, symbol):
    rows = re.findall(
        r"<th[^>]*>\s*(\d{1,2}-[A-Z]{3}-\d{2})\s*</th>\s*<td[^>]*>\s*(ND|[0-9.]+)\s*</td>",
        html,
    )
    if len(rows) < 1000:
        raise ValueError("Unexpected source table; refuse partial import")
    records, missing = [], 0
    for date, value in rows:
        day = datetime.strptime(date, "%d-%b-%y").strftime("%Y-%m-%d")
        if not "2015-01-01" <= day <= "2025-12-31":
            continue
        if value == "ND":
            missing += 1
            continue
        records.append({"Date": day, "Close": float(value), "Symbol": symbol})
    frame = pd.DataFrame(records)
    if (
        len(frame) < 2500
        or frame.Date.duplicated().any()
        or not frame.Date.is_monotonic_increasing
        or (frame.Close <= 0).any()
    ):
        raise ValueError("Invalid imported observations")
    return frame, missing


def save_snapshot(responses):
    parts, sources = [], []
    for symbol, code in SOURCES.items():
        raw = responses[code]
        frame, missing = parse_rates(raw.decode(), symbol)
        parts.append(frame)
        sources.append(
            {
                "symbol": symbol,
                "url": f"https://www.federalreserve.gov/releases/h10/hist/dat00_{code}.htm",
                "source_response_sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(frame),
                "missing_ND_rows_removed": missing,
                "unit": "U.S. dollars per euro"
                if code == "eu"
                else "U.S. dollars per British pound",
            }
        )
    raw = (
        pd.concat(parts, ignore_index=True)
        .to_csv(index=False, lineterminator="\n")
        .encode()
    )
    (ROOT / "data/official_fx.csv").write_bytes(raw)
    manifest = {
        "publisher": "Board of Governors of the Federal Reserve System",
        "release": "H.10",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval": ["2015-01-01", "2025-12-31"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "sources": sources,
        "reuse": "Public domain unless otherwise indicated; source attribution required",
        "rights_url": "https://www.federalreserve.gov/disclaimer.htm",
        "definition_url": "https://www.federalreserve.gov/releases/h10/about.htm",
        "transformations": "ND omitted without forward fill; returns span consecutive available observations, including holiday gaps. No equity corporate actions apply. Reference quotes, not executable prices. Latest historical vintage, not point-in-time vintage.",
    }
    (ROOT / "data/official_fx_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main():
    responses = {}
    for code in SOURCES.values():
        response = httpx.get(
            f"https://www.federalreserve.gov/releases/h10/hist/dat00_{code}.htm",
            timeout=40,
            follow_redirects=True,
        )
        response.raise_for_status()
        responses[code] = response.content
    save_snapshot(responses)


if __name__ == "__main__":
    main()
