"""Tests for time series modules (standard-library unittest)."""

import unittest

import numpy as np
import pandas as pd

from src.time_series.forecaster import forecast_returns, forecast_volatility
from src.time_series.preprocessor import prepare_prices


class TimeSeriesTests(unittest.TestCase):
    def test_prepare_prices(self):
        np.random.seed(7)
        dates = pd.date_range("2023-01-01", periods=100)
        close = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = pd.DataFrame({"Date": dates, "Close": close})
        out = prepare_prices(df, price_col="Close")
        self.assertTrue(
            {"log_return", "ewma_volatility", "trend", "residual"}.issubset(out.columns)
        )
        self.assertEqual(len(out), 100)

    def test_forecast_returns(self):
        np.random.seed(8)
        returns = np.random.normal(0.001, 0.01, 50)
        fc = forecast_returns(returns, horizon=10, method="ar")
        self.assertIn("mean_daily_return", fc)
        self.assertEqual(len(fc["mean_daily_return"]), 10)
        self.assertIn("cumulative_return", fc)

    def test_forecast_volatility(self):
        np.random.seed(9)
        returns = np.random.normal(0, 0.02, 100)
        vfc = forecast_volatility(returns, horizon=15)
        self.assertIn("daily_volatility", vfc)
        self.assertEqual(len(vfc["daily_volatility"]), 15)
        self.assertGreater(vfc["annualized_volatility"], 0)


if __name__ == "__main__":
    unittest.main()
