"""Tests for portfolio risk metrics calculation (standard-library unittest)."""
import unittest
import numpy as np
from src.risk.metrics import risk_metrics

class RiskMetricsTests(unittest.TestCase):
    def test_risk_metrics_basic(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)
        res = risk_metrics(returns, risk_free_rate=0.02)
        expected = {"var_95", "cvar_95", "var_99", "cvar_99", "sharpe",
                    "sortino", "max_drawdown", "annualized_return"}
        self.assertTrue(expected.issubset(res))
        self.assertGreaterEqual(res["cvar_95"], res["var_95"])
        self.assertGreaterEqual(res["cvar_99"], res["var_99"])
        self.assertLessEqual(res["max_drawdown"], 0.0)
        self.assertEqual(res["observations"], 500)

    def test_risk_metrics_invalid_input(self):
        with self.assertRaises(ValueError):
            risk_metrics([0.01])
        with self.assertRaises(ValueError):
            risk_metrics([np.nan, np.inf])

if __name__ == "__main__":
    unittest.main()
