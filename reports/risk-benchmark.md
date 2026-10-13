# Attributed reference-rate risk benchmark

Frozen H.10 EURUSD / GBPUSD snapshot. Evaluation: 2020–2025. Previous observations only; 252-observation window, EWMA decay 0.94, 95% loss VaR, zero conditional mean. No tuning after viewing results.

| Pair | Method | Exceptions | Rate | Exact coverage p | Quantile loss | QLIKE |
|---|---|---:|---:|---:|---:|---:|
| EURUSD | rolling_normal | 82/1499 | 5.47% | 0.4063 | 0.00053447 | -9.654102371463297 |
| EURUSD | ewma_normal | 81/1499 | 5.40% | 0.4766 | 0.00051125 | -9.775194732866403 |
| EURUSD | historical | 90/1499 | 6.00% | 0.0755 | 0.00053040 | n/a |
| GBPUSD | rolling_normal | 72/1499 | 4.80% | 0.7672 | 0.00066635 | -9.367047698638794 |
| GBPUSD | ewma_normal | 78/1499 | 5.20% | 0.7219 | 0.00061778 | -9.507734709754573 |
| GBPUSD | historical | 79/1499 | 5.27% | 0.6352 | 0.00064882 | n/a |

Expected exception rate: 5%. Smaller quantile loss / QLIKE is better; QLIKE has a scale-dependent offset and uses squared returns as a noisy variance proxy. Historical VaR has no variance forecast. Exact binomial coverage assumes independent exceptions; transition counts and an asymptotic independence diagnostic are included in the JSON, with sparse-sample limitations. Non-rejection does not establish safety or profitability.

The snapshot is attributed and integrity checked, not independently corroborated with a second publisher. Reference rates are not executable prices. No OHLC or volume is fabricated. ND holidays are omitted without filling; returns span consecutive available observations. Historical revisions are not reconstructed. Annual tables and every forecast are included in risk-benchmark.json. The 2025 slice is a reporting slice, not an untouched holdout.
