# Review improvement validation — 2026-10-13

Local Windows validation: 32 tests; actual uvicorn HTTP checks on both supplied instruments; isolated desktop and mobile browser checks. No paid model endpoint used in this run.

## Chronological comparison

| Instrument | AR1 MAE | Zero MAE | Historical mean MAE | Beats both |
|---|---:|---:|---:|---|
| AAPL | 0.01508595 | 0.01498899 | 0.01484311 | False |
| SPY | 0.00659703 | 0.00651232 | 0.00645719 | False |

Daily duplicate rejection, input hash sensitivity, causal predictions, error-score reconciliation and chart warm-up have regression coverage. Sample provenance remains unverified. Vision remains numeric OHLC rules, not learned image interpretation. The cached Qwen3-0.6B model was rerun successfully against both instruments and the multipart upload through the real HTTP service; no paid endpoint was used. Current raw model responses are in live-local.json. See live-deterministic.json for current raw responses.

Independent Ubuntu CI also passed on code commit 1ed13a844774a6380d399e95b43c16b8ccb98b67: https://github.com/saintedwind/multimodal-risk-copilot/actions/runs/38416623870 . The workflow includes tests, loopback HTTP, Chromium desktop/mobile and Docker API execution. Artifact SHA-256: 456070df9465b7eda30e767dda9b09882d5f440e0642fc4afaad150337837d62.
