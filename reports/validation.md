# Validation ledger / 2026-10-13

Python 3.12, Windows CPU. Actual loopback HTTP service and a real local model were run, not mocked remote calls. Data provenance remains unverified; live testing does not mean a live market feed.

## Observed checks

- **27 passing tests**, including original unit tests, independent GBM expected-value tolerance, known loss quantile/drawdown reconciliation, return conventions, rejected non-finite input, seed reproducibility, CSV/date/instrument validation, concurrency capacity, and a real Streamlit AppTest execution.
- Actual uvicorn HTTP checks succeeded: separate AAPL/SPY analysis, real PNG decode/hash, repeated seed equality, oversized simulation rejection, malformed CSV rejection, mixed-instrument rejection, and selected-instrument multipart upload.
- Actual Qwen3-0.6B Q8_0 / llama.cpp b11476 selected valid supplied observations for both instrument responses. Their recorded local-model usage totals 428 tokens; no paid API was called. Raw [deterministic](live-deterministic.json) and [local](live-local.json) responses record values, status, usage and elapsed times. One additional CSV upload also exercised local selection; its usage is not included in this sum.
- Expanding-window one-step AR(1) evaluated the last 100 observations per instrument, using only each preceding prefix to fit. No profitable trading strategy, transaction-cost result or independent market benchmark is represented.

| Instrument | AR(1) MAE, return fraction | Zero-return MAE | Beats baseline |
|---|---:|---:|---|
| AAPL | 0.015086 | 0.014989 | No |
| SPY | 0.006597 | 0.006512 | No |

The baseline wins on both samples. Do not describe this implementation as having demonstrated predictive advantage. The chronological comparison is a development diagnostic on supplied data, not independently validated market evidence.

Model SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`; official Qwen revision `23749fefcc72300e3a2ad315e1317431b06b590a`. CPU context 4096, 4 threads, one slot, 64 output tokens per selection. Owned test processes were stopped. Model weights are excluded.

## Review corrections

The initial code mixed log returns into simple-return wealth calculations and fitted GBM drift without its variance correction. Multiple instruments were concatenated. It returned invented image-only diagnoses, fixed pattern confidence numbers and mislabeled EWMA as GARCH. Paid reporting could activate simply because an API key existed. These paths are corrected or explicitly unsupported; reports no longer prescribe trades or hedges. Input/memory budgets and truthful method/fallback statuses were added. API and dashboard now share one service; Windows-specific temporary-path failures were removed.

## Limits

No trained visual inference, independently verified price provenance, multi-asset weights, institutional deployment/security controls or demonstrated forecast advantage. Pattern rules are heuristics with null confidence. Images are truly decoded and hashed but that is not semantic chart recognition. Docker execution, load testing and remote deployment are not yet verified. Browser CI result and reviewed screenshots will be appended after actual execution.
