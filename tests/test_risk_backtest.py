import numpy as np
import pytest
from scipy.stats import binomtest
from src.risk.backtest import backtest_risk, coverage_diagnostics
from src.data_sources import load_official
from fastapi.testclient import TestClient
from src.app.api import app


def test_forecasts_do_not_use_target_or_future():
    x = np.random.default_rng(19).normal(0, 0.01, 310)
    original = backtest_risk(x)
    changed = x.copy()
    changed[280:] = 0.2
    altered = backtest_risk(changed)
    for name in original["predictions"]:
        assert (
            original["predictions"][name]["var"][:29]
            == altered["predictions"][name]["var"][:29]
        )
    assert original["predictions"]["rolling_normal"]["variance"][0] == pytest.approx(
        np.mean(x[:252] ** 2)
    )
    q = np.asarray(original["predictions"]["historical"]["var"])
    loss = -x[252:]
    score = np.mean(np.where(loss >= q, 0.95 * (loss - q), 0.05 * (q - loss)))
    assert original["models"]["historical"]["mean_quantile_loss"] == pytest.approx(
        score
    )


def test_coverage_counts_zero_tails_and_exact_probability():
    c = coverage_diagnostics([0, 0, 1, 1, 0], 0.95)
    assert c["transitions_00_01_10_11"] == [1, 1, 1, 1]
    assert c["exact_binomial_p_value"] == pytest.approx(binomtest(2, 5, 0.05).pvalue)
    zero = coverage_diagnostics([0] * 30, 0.95)
    assert zero["independence"] is None
    flat = backtest_risk([0.0] * 300)
    assert np.isfinite(flat["models"]["ewma_normal"]["mean_qlike"])


def test_official_snapshot_and_real_api_contract():
    data = load_official()
    assert set(data.Symbol) == {"EURUSD", "GBPUSD"}
    assert len(data) == 5496
    assert not data.duplicated(["Symbol", "Date"]).any()
    with TestClient(app) as client:
        response = client.get("/analyze/reference?symbol=EURUSD")
        assert response.status_code == 200
        result = response.json()
        assert result["risk_metrics"]["observations"] == 2747
        assert result["risk_evaluation"]["status"] == "evaluated"
        assert result["source_manifest"]["release"] == "H.10"
        assert client.get("/analyze/reference?symbol=UNKNOWN").status_code == 422


def test_short_history_is_not_presented_as_backtested():
    assert backtest_risk([0.01] * 260)["status"] == "insufficient_history"
    with pytest.raises(ValueError):
        backtest_risk([0.01] * 300, start_index=200)


def test_invalid_exception_indicators_are_not_truncated():
    with pytest.raises(ValueError):
        coverage_diagnostics([0, 0.5, 1], 0.95)


def test_corrupted_snapshot_is_rejected(tmp_path, monkeypatch):
    import json
    import src.data_sources as source

    (tmp_path / "data").mkdir()
    manifest = load_official().attrs["source_manifest"]
    (tmp_path / "data/official_fx_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "data/official_fx.csv").write_bytes(b"tampered")
    monkeypatch.setattr(source, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        source.load_official()
