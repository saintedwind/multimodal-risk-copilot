"""Chronological development evaluation; each prediction sees only earlier returns."""

import numpy as np
from src.risk.metrics import _arr
from src.time_series.forecaster import forecast_returns


def evaluate_forecast(returns, test_size=100, min_train=60):
    x = _arr(returns)
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, int)
        or not 1 <= test_size <= 500
    ):
        raise ValueError("test_size must be 1..500")
    if isinstance(min_train, bool) or not isinstance(min_train, int) or min_train < 5:
        raise ValueError("min_train must be at least five")
    start = max(min_train, len(x) - test_size)
    if start >= len(x):
        return {"status": "insufficient_history", "minimum_observations": min_train + 1}
    predictions = {"ar1": [], "zero": [], "historical_mean": []}
    actual = x[start:]
    for i in range(start, len(x)):
        predictions["ar1"].append(forecast_returns(x[:i], 1)["mean_daily_return"][0])
        predictions["zero"].append(0.0)
        predictions["historical_mean"].append(float(x[:i].mean()))
    scores = {}
    for name, values in predictions.items():
        errors = np.asarray(values) - actual
        scores[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        }
    return {
        "status": "evaluated",
        "protocol": "expanding_window_one_step",
        "training_observations_at_start": start,
        "test_observations": len(actual),
        "scores": scores,
        "beats_both_baselines_mae": scores["ar1"]["mae"]
        < min(scores["zero"]["mae"], scores["historical_mean"]["mae"]),
        "scope": "development comparison on supplied data; not an independent market benchmark",
        "predictions": predictions,
        "actual": actual.tolist(),
    }
