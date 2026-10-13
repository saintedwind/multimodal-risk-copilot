# Reference-rate release validation — 2026-10-13

Version 0.3.0 adds a frozen, attributed H.10 snapshot and retrospective risk evaluation. Data source rights, quote units, retrieval time and checksums are recorded in data/official_fx_manifest.json. No account, subscription or paid endpoint was required.

## Evidence

- 38 local tests passed: forecast causality, loss-score reconciliation, exception transition counts, exact binomial probability, zero-variance behavior, invalid indicator rejection, corrupted snapshot rejection, both reference API contracts and existing numeric/image/UI checks.
- Real loopback uvicorn requests succeeded for EURUSD, GBPUSD, legacy AAPL/SPY and selected multipart CSV. Resource limits, invalid uploads and mixed symbols were checked. Current deterministic responses are in live-deterministic.json.
- Isolated Windows Chrome desktop/mobile checks passed. Risk-table and source-label rendering were visually inspected. Screenshots are desk-desktop.png, desk-mobile.png and desk-risk-evaluation.png.
- Cached Qwen3-0.6B Q8_0 successfully selected fixed report observations through the real HTTP service for EURUSD and GBPUSD, as well as the legacy checks. Each reference call generated 26 completion tokens on the local CPU. Raw model evidence is in live-local.json; it predates the final date-label/display-only cleanup. No paid endpoint was called.
- The offline benchmark exports every chronological forecast, annual slices and source digest in risk-benchmark.json. Parameters were fixed before the first benchmark run; no parameter search or tuning followed inspection.

## Interpretation

For the fixed 2020–2025 interval, EWMA had lower QLIKE than rolling variance for both pairs, and in 11 of the 12 pair/year slices. Aggregate quantile loss was lower than both the historical and rolling-normal VaR baselines for each pair. Exception rates for EWMA were 5.40% (EURUSD) and 5.20% (GBPUSD), against a 5% expected rate. These are descriptive results on two correlated FX pairs, not a statistical superiority claim or proof of future calibration/profitability.

The dashboard evaluates all eligible history after its first 252 returns; the repository benchmark separately fixes 2020–2025. Both explicitly report their scope. Squared daily returns are only a noisy variance proxy. Historical reference rates may be revised; the frozen latest-vintage data is not a point-in-time dataset or executable market feed. Legacy OHLC remains unverified; reference OHLC/volume is not fabricated. Coverage non-rejection does not certify a risk model. The 2025 slice is not an untouched prospective holdout.

## Independent execution

[Ubuntu CI run 38423530420](https://github.com/saintedwind/multimodal-risk-copilot/actions/runs/38423530420) succeeded on code commit `b844a598942632c9d1e339b7afee5d342919e877`, including tests, offline benchmark, actual HTTP, Chromium desktop/mobile, Docker build and the live reference endpoint inside the container. Subsequent documentation-only commits do not change this tested code. Artifact SHA-256: `bb6a511e054bc2350e5251104b97a6886953e3ce3cf8819fe355a839900679c4`. The original published snapshot's byte checksum was independently reconciled against the source manifest after upload.
