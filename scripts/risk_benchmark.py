"""Reproduce the fixed 2020–2025 reference-rate risk comparison offline."""

import json
from pathlib import Path
import pandas as pd
from src.data_sources import load_official
from src.risk.backtest import backtest_risk

ROOT = Path(__file__).resolve().parents[1]


def main():
    frame = load_official()
    rows = []
    for symbol in sorted(frame.Symbol.unique()):
        prices = frame[frame.Symbol == symbol].reset_index(drop=True)
        returns = prices.Close.pct_change().iloc[1:].to_numpy()
        dates = pd.to_datetime(prices.Date.iloc[1:]).reset_index(drop=True)
        start = int((dates < "2020-01-01").sum())
        result = backtest_risk(returns, start_index=start)
        assert result["status"] == "evaluated"
        models = result.pop("models")
        result["dates"] = dates.iloc[start:].dt.strftime("%Y-%m-%d").tolist()
        yearly = {}
        for year in range(2020, 2026):
            stop = int((dates <= f"{year}-12-31").sum())
            begin = int((dates < f"{year}-01-01").sum())
            yearly[str(year)] = backtest_risk(returns[:stop], start_index=begin)[
                "models"
            ]
        rows.append(
            {"symbol": symbol, "models": models, "yearly": yearly, "evidence": result}
        )
    report = {
        "source_manifest": frame.attrs["source_manifest"],
        "protocol": {
            "evaluation_interval": ["2020-01-01", "2025-12-31"],
            "window": 252,
            "decay": 0.94,
            "alpha": 0.95,
            "selection": "fixed methods and parameters; no search or tuning",
            "scope": "retrospective chronological evaluation on latest vintage; not untouched prospective holdout; two correlated FX pairs",
        },
        "results": rows,
    }
    (ROOT / "reports/risk-benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    text = [
        "# Attributed reference-rate risk benchmark",
        "",
        "Frozen H.10 EURUSD / GBPUSD snapshot. Evaluation: 2020–2025. Previous observations only; 252-observation window, EWMA decay 0.94, 95% loss VaR, zero conditional mean. No tuning after viewing results.",
        "",
        "| Pair | Method | Exceptions | Rate | Exact coverage p | Quantile loss | QLIKE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for name, v in row["models"].items():
            c = v["coverage"]
            text.append(
                f"| {row['symbol']} | {name} | {c['exceptions']}/{c['observations']} | {c['exception_rate']:.2%} | {c['exact_binomial_p_value']:.4f} | {v['mean_quantile_loss']:.8f} | {v.get('mean_qlike', 'n/a')} |"
            )
    text += [
        "",
        "Expected exception rate: 5%. Smaller quantile loss / QLIKE is better; QLIKE has a scale-dependent offset and uses squared returns as a noisy variance proxy. Historical VaR has no variance forecast. Exact binomial coverage assumes independent exceptions; transition counts and an asymptotic independence diagnostic are included in the JSON, with sparse-sample limitations. Non-rejection does not establish safety or profitability.",
        "",
        "The snapshot is attributed and integrity checked, not independently corroborated with a second publisher. Reference rates are not executable prices. No OHLC or volume is fabricated. ND holidays are omitted without filling; returns span consecutive available observations. Historical revisions are not reconstructed. Annual tables and every forecast are included in risk-benchmark.json. The 2025 slice is a reporting slice, not an untouched holdout.",
    ]
    (ROOT / "reports/risk-benchmark.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    print("\n".join(text[:13]))


if __name__ == "__main__":
    main()
