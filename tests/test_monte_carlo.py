"""Tests for Monte Carlo simulation engine (standard-library unittest)."""

import unittest

import numpy as np

from src.risk.monte_carlo import simulate_monte_carlo


class MonteCarloTests(unittest.TestCase):
    def test_simulate_monte_carlo_gbm(self):
        res = simulate_monte_carlo(
            initial_price=100.0,
            mu=0.08,
            sigma=0.20,
            days=252,
            n_paths=1000,
            model="gbm",
            seed=42,
        )
        self.assertTrue(
            {"paths", "terminal_prices", "returns", "summary"}.issubset(res)
        )
        self.assertEqual(res["paths"].shape, (1000, 253))
        self.assertTrue(np.allclose(res["paths"][:, 0], 100.0))
        summary = res["summary"]
        self.assertEqual(summary["n_paths"], 1000)
        self.assertEqual(summary["days"], 252)
        self.assertGreaterEqual(summary["loss_probability"], 0.0)
        self.assertLessEqual(summary["loss_probability"], 1.0)

    def test_simulate_monte_carlo_jump_diffusion(self):
        res = simulate_monte_carlo(
            initial_price=100.0,
            mu=0.05,
            sigma=0.15,
            days=100,
            n_paths=500,
            model="jump_diffusion",
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_sigma=0.1,
            seed=123,
        )
        self.assertEqual(res["paths"].shape, (500, 101))
        self.assertEqual(len(res["terminal_prices"]), 500)

    def test_simulate_monte_carlo_exceptions(self):
        with self.assertRaises(ValueError):
            simulate_monte_carlo(initial_price=-10.0, mu=0.05, sigma=0.2)
        with self.assertRaises(ValueError):
            simulate_monte_carlo(initial_price=100.0, mu=0.05, sigma=-0.2)
        with self.assertRaises(ValueError):
            simulate_monte_carlo(
                initial_price=100.0, mu=0.05, sigma=0.2, model="invalid_model"
            )


if __name__ == "__main__":
    unittest.main()
