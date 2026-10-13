import numpy as np
import pandas as pd
import pytest
from src.time_series.evaluation import evaluate_forecast
from src.time_series.preprocessor import prepare_prices
from src.app.service import analyze_returns


def test_evaluation_is_causal_and_reconciles():
    x = np.random.default_rng(42).normal(0, 0.01, 100)
    original = evaluate_forecast(x, test_size=40)
    changed = x.copy()
    changed[-1] = 0.25
    altered = evaluate_forecast(changed, test_size=40)
    assert original["predictions"] == altered["predictions"]
    for name, values in original["predictions"].items():
        errors = np.asarray(values) - x[60:]
        assert original["scores"][name]["mae"] == pytest.approx(np.abs(errors).mean())
    assert original["predictions"]["historical_mean"][0] == pytest.approx(x[:60].mean())


def test_daily_contract_rejects_intraday_duplicates():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=6, freq="h"),
            "Close": range(100, 106),
        }
    )
    with pytest.raises(ValueError, match="calendar day"):
        prepare_prices(frame)


def test_audit_hash_changes_with_input_and_short_history_is_explicit():
    x = [0.01, -0.02, 0.03, -0.01, 0.02]
    a = analyze_returns(x, n_paths=500)
    b = analyze_returns(x, n_paths=500)
    assert a["audit"] == b["audit"]
    assert a["forecast_evaluation"]["status"] == "insufficient_history"
    assert analyze_returns(x + [0.02], n_paths=500)["audit"] != a["audit"]


def test_chart_indicators_use_history_before_visible_window(monkeypatch):
    import io
    import matplotlib.pyplot as plt
    from src.vision.chart_renderer import render_chart

    close = np.linspace(100, 140, 180)
    frame = pd.DataFrame(
        {"Open": close, "Close": close, "High": close + 1, "Low": close - 1}
    )
    captured = []
    original = plt.subplots

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture)
    render_chart(frame, io.BytesIO())
    ma20 = captured[0][1][0].lines[0].get_ydata()
    assert np.isfinite(ma20).all()
    assert ma20[0] == pytest.approx(close[41:61].mean())
