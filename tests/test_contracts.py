import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app.api import app
from src.app.service import analyze_prices, analyze_returns
from src.risk.metrics import risk_metrics
from src.risk.monte_carlo import simulate_monte_carlo
from src.time_series.preprocessor import prepare_prices
from src.vision.chart_renderer import compute_rsi
from src.vision.pattern_agent import analyze_chart


def test_simple_return_reconciliation():
    df = pd.DataFrame({"Close": [100, 110, 99, 108.9, 119.79, 107.811]})
    out = analyze_prices(df, n_paths=100, horizon_days=10)
    assert out["risk_metrics"]["annualized_return"] == pytest.approx(
        (107.811 / 100) ** (252 / 5) - 1
    )
    assert out["risk_metrics"]["max_drawdown"] == pytest.approx(-0.1)
    assert out["report_status"] == "deterministic"


def test_gbm_drift_fit_reconciles_log_mean():
    x = [0.1, -0.1, 0.03, -0.02, 0.01, 0.04]
    s = analyze_returns(x, n_paths=100, horizon_days=10)["monte_carlo_summary"]
    assert s["mu"] - 0.5 * s["sigma"] ** 2 == pytest.approx(np.log1p(x).mean() * 252)


@pytest.mark.parametrize(
    "x", [[0.1, np.nan, 0.2], [0.1, np.inf, 0.2], [0.1, -1, 0.2], [[0.1, 0.2]]]
)
def test_invalid_returns_not_silently_dropped(x):
    with pytest.raises(ValueError):
        risk_metrics(x)


def test_zero_volatility_ratios_are_null():
    m = risk_metrics([0.0] * 30)
    assert m["sharpe"] is None and m["sortino"] is None and m["calmar"] is None


def test_known_quantile_and_tail_mean():
    m = risk_metrics([-0.1, -0.05, 0.0, 0.02, 0.03])
    assert m["var_95"] == pytest.approx(0.09)
    assert m["cvar_95"] == pytest.approx(0.1)


def test_chronology_and_symbols():
    for df in [
        pd.DataFrame({"Close": [1] * 6, "Date": ["2024-01-01"] * 6}),
        pd.DataFrame({"Close": [1] * 6, "Symbol": ["A", "B"] * 3}),
    ]:
        with pytest.raises(ValueError):
            prepare_prices(df)


def test_gbm_independent_analytic_moments():
    r = simulate_monte_carlo(100, 0.08, 0.2, days=100, n_paths=10000, seed=1337)
    terminal = r["terminal_prices"]
    expected = 100 * np.exp(0.08 * 100 / 252)
    se = 100 * np.exp(0.08 * 100 / 252) * np.sqrt(np.expm1(0.2**2 * 100 / 252) / 10000)
    assert abs(terminal.mean() - expected) < 4 * se


def test_resource_budget_and_seed():
    assert simulate_monte_carlo(100, 0, 0.2, seed=42)["paths"].shape == (1000, 253)
    with pytest.raises(ValueError):
        simulate_monte_carlo(100, 0, 0.2, days=504, n_paths=10000)
    a = simulate_monte_carlo(100, 0.1, 0.2, days=10, n_paths=100, seed=3)
    b = simulate_monte_carlo(100, 0.1, 0.2, days=10, n_paths=100, seed=3)
    assert np.array_equal(a["paths"], b["paths"])


def test_image_only_does_not_fabricate_patterns():
    r = analyze_chart("")
    assert (
        r["patterns_detected"] == []
        and r["multimodal_source"] == "image_only_not_supported"
    )


def test_valid_png_structure_with_invalid_pixels_is_rejected(tmp_path):
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", b"not-a-zlib-stream")
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "corrupt.png"
    path.write_bytes(raw)
    with pytest.raises(OSError):
        analyze_chart(str(path))


def test_flat_rsi_is_neutral():
    assert compute_rsi(pd.Series([100.0] * 30)).iloc[-1] == 50


def test_http_contract_and_csv():
    with TestClient(app) as c:
        assert c.get("/health").json()["paid_api_enabled"] is False
        body = {
            "returns": [0.01, -0.02, 0.03, -0.01, 0.02],
            "horizon_days": 10,
            "n_paths": 100,
            "seed": 42,
        }
        a = c.post("/analyze/metrics", json=body)
        b = c.post("/analyze/metrics", json=body)
        assert a.status_code == 200 and a.json() == b.json()
        assert (
            c.post(
                "/analyze/metrics", json={**body, "n_paths": 10000, "horizon_days": 504}
            ).status_code
            == 422
        )

        df = pd.DataFrame({"Close": [100, 101, 99, 102, 100, 101]})
        r = c.post(
            "/analyze/full",
            files={"file": ("test.csv", df.to_csv(index=False).encode(), "text/csv")},
        )
        assert (
            r.status_code == 200
            and r.json()["patterns"]["multimodal_source"] == "ohlc_not_available"
        )
        assert (
            c.post(
                "/analyze/full", files={"file": ("bad.csv", b"Close\nnope", "text/csv")}
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/analyze/full",
                files={"file": ("large.csv", b"x" * 2_000_001, "text/csv")},
            ).status_code
            == 413
        )


def test_api_selects_one_instrument():
    with TestClient(app) as c:
        a = c.get("/analyze/sample?symbol=SPY")
        b = c.get("/analyze/sample?symbol=AAPL")
        assert a.status_code == b.status_code == 200
        assert a.json()["assumptions"]["instrument"] == "SPY"
        assert a.json()["risk_metrics"]["observations"] == 329
        assert c.get("/analyze/sample?symbol=missing").status_code == 400


def test_busy_capacity_returns_explicit_status():
    from src.app.api import slots

    slots.acquire()
    slots.acquire()
    try:
        with TestClient(app) as c:
            assert c.get("/analyze/sample").status_code == 503
    finally:
        slots.release()
        slots.release()


def test_secret_presence_does_not_trigger_paid_calls(monkeypatch):
    from src.llm_engine.risk_agent import RiskCopilotAgent

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert RiskCopilotAgent().provider == "deterministic"


def test_preprocessor_prefix_is_causal():
    df = pd.DataFrame({"Close": np.linspace(100, 120, 30)})
    a = prepare_prices(df.iloc[:20])
    b = prepare_prices(df)
    assert np.allclose(a.trend, b.trend.iloc[:20])
