"""Real HTTP analysis, real image decoding, and chronological forecast comparison."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from src.time_series.forecaster import forecast_returns
from src.vision.chart_renderer import render_chart
from src.vision.pattern_agent import analyze_chart

ROOT = Path(__file__).resolve().parents[1]


def main():
    mode = os.getenv("REPORT_PROVIDER", "deterministic")
    (ROOT / "reports").mkdir(exist_ok=True)
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8000))
    finally:
        s.close()
    with (ROOT / "reports" / f"live-{mode}.log").open("w") as log:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.app.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url="http://127.0.0.1:8000", timeout=60, trust_env=False
            ) as c:
                deadline = time.monotonic() + 90
                while True:
                    if p.poll() is not None:
                        raise RuntimeError("API exited")
                    try:
                        c.get("/health").raise_for_status()
                        break
                    except httpx.HTTPError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("API startup")
                        time.sleep(0.5)
                data = pd.read_csv(ROOT / "data/sample_prices.csv")
                official_rows = []
                for pair in ["EURUSD", "GBPUSD"]:
                    response = c.get("/analyze/reference", params={"symbol": pair})
                    response.raise_for_status()
                    value = response.json()
                    assert value["risk_metrics"]["observations"] == 2747
                    assert value["risk_evaluation"]["status"] == "evaluated"
                    assert value["source_manifest"]["release"] == "H.10"
                    official_rows.append({"symbol": pair, "response": value})
                rows = []
                backtests = []
                for symbol in sorted(data.Symbol.unique()):
                    t = time.perf_counter()
                    r = c.get("/analyze/sample", params={"symbol": symbol})
                    r.raise_for_status()
                    result = r.json()
                    assert result["risk_metrics"][
                        "return_type"
                    ] == "daily_simple" and result["report_status"] in {
                        "deterministic",
                        "local_selection",
                    }
                    frame = data[data.Symbol == symbol].reset_index(drop=True)
                    path = ROOT / "reports" / f"{symbol.lower()}-chart.png"
                    render_chart(frame, str(path))
                    image = analyze_chart(str(path), frame)
                    assert (
                        image["image_artifact"]["status"] == "decoded_and_verified_only"
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "elapsed_seconds": round(time.perf_counter() - t, 3),
                            "response": result,
                            "image_validation": image["image_artifact"],
                        }
                    )
                    returns = frame.Close.pct_change().dropna().to_numpy()
                    errors = []
                    zero = []
                    for i in range(max(60, len(returns) - 100), len(returns)):
                        pred = forecast_returns(returns[:i], 1)["mean_daily_return"][0]
                        errors.append(abs(pred - returns[i]))
                        zero.append(abs(returns[i]))
                    backtests.append(
                        {
                            "symbol": symbol,
                            "method": "expanding-window one-step AR1 vs zero-return baseline",
                            "test_observations": len(errors),
                            "ar1_mae": float(np.mean(errors)),
                            "zero_baseline_mae": float(np.mean(zero)),
                            "outperforms_baseline": bool(
                                np.mean(errors) < np.mean(zero)
                            ),
                            "independent_market_provenance": False,
                        }
                    )
                payload = {
                    "returns": [0.01, -0.02, 0.03, -0.01, 0.02],
                    "horizon_days": 30,
                    "n_paths": 500,
                    "seed": 42,
                }
                a = c.post("/analyze/metrics", json=payload)
                a.raise_for_status()
                assert a.json() == c.post("/analyze/metrics", json=payload).json()
                assert (
                    c.post(
                        "/analyze/metrics",
                        json={**payload, "horizon_days": 504, "n_paths": 10000},
                    ).status_code
                    == 422
                )
                assert (
                    c.post(
                        "/analyze/full",
                        files={"file": ("bad.csv", b"Close\nnope", "text/csv")},
                    ).status_code
                    == 400
                )
                raw = (ROOT / "data/sample_prices.csv").read_bytes()
                r = c.post(
                    "/analyze/full",
                    data={"symbol": "SPY"},
                    files={"file": ("sample.csv", raw, "text/csv")},
                )
                r.raise_for_status()
                assert (
                    c.post(
                        "/analyze/full", files={"file": ("sample.csv", raw, "text/csv")}
                    ).status_code
                    == 400
                )
                report = {
                    "date": "2026-10-13",
                    "transport": "actual loopback HTTP / uvicorn",
                    "provider": mode,
                    "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_provenance": "unverified provided sample; no live market feed",
                    "rows": rows,
                    "official_reference_rows": official_rows,
                    "backtests": backtests,
                    "checks": [
                        "both instruments analyzed separately",
                        "real PNG decoded",
                        "same seed reproducible",
                        "resource budget rejected",
                        "malformed CSV rejected",
                        "mixed symbols rejected unless selected",
                        "multipart CSV succeeded",
                    ],
                }
                (ROOT / "reports" / f"live-{mode}.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "provider": mode,
                            "checks": report["checks"],
                            "backtests": backtests,
                        },
                        indent=2,
                    )
                )
        finally:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


if __name__ == "__main__":
    main()
