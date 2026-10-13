"""Vectorized GBM and Merton jump-diffusion scenarios."""

from __future__ import annotations

import numpy as np


def simulate_monte_carlo(
    initial_price,
    mu,
    sigma,
    days=252,
    n_paths=1000,
    model="gbm",
    jump_intensity=0.0,
    jump_mean=0.0,
    jump_sigma=0.0,
    seed=None,
):
    if initial_price <= 0 or sigma < 0 or days < 1 or n_paths < 1:
        raise ValueError("invalid price, volatility, horizon, or path count")
    if model not in {"gbm", "jump_diffusion"}:
        raise ValueError("model must be gbm or jump_diffusion")
    if jump_intensity < 0 or jump_sigma < 0:
        raise ValueError("jump parameters cannot be negative")
    values = [initial_price, mu, sigma, jump_intensity, jump_mean, jump_sigma]
    if (
        not np.isfinite(values).all()
        or type(days) is not int
        or type(n_paths) is not int
        or days > 504
        or n_paths > 10000
        or days * n_paths > 2_000_000
    ):
        raise ValueError("Non-finite inputs or simulation budget exceeded")
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal((n_paths, days))
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
    if model == "jump_diffusion" and jump_intensity:
        counts = rng.poisson(jump_intensity * dt, (n_paths, days))
        jumps = counts * jump_mean + jump_sigma * np.sqrt(counts) * rng.standard_normal(
            (n_paths, days)
        )
        increments += (
            jumps - jump_intensity * (np.exp(jump_mean + 0.5 * jump_sigma**2) - 1) * dt
        )
    paths = initial_price * np.exp(np.cumsum(increments, axis=1))
    paths = np.concatenate([np.full((n_paths, 1), initial_price), paths], axis=1)
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError("Simulation outside numerical domain")
    returns = paths[:, -1] / initial_price - 1
    return {
        "paths": paths,
        "terminal_prices": paths[:, -1],
        "returns": returns,
        "summary": {
            "mean_return": float(returns.mean()),
            "median_return": float(np.median(returns)),
            "loss_probability": float(np.mean(returns < 0)),
            "model": model,
            "n_paths": n_paths,
            "days": days,
            "seed": seed,
            "mu": float(mu),
            "sigma": float(sigma),
            "jump_intensity": float(jump_intensity),
            "jump_mean": float(jump_mean),
            "jump_sigma": float(jump_sigma),
        },
    }
