"""Prompt templates for Risk Copilot LLM synthesized reports."""
from __future__ import annotations

RISK_EXECUTIVE_SYSTEM_PROMPT = """You are an elite Chief Risk Officer (CRO) and Quantitative Financial Analyst.
Your task is to synthesize multimodal quantitative risk data, time-series forecasts, and technical chart patterns into an executive-grade risk report.
Maintain rigorous institutional standards. Highlight tail risk, Value at Risk (VaR), Conditional VaR (CVaR), regime shifts, and actionable hedging recommendations.
Format your output cleanly in Markdown with executive summary, quantitative findings, stress test insights, and mitigating strategies.
"""

RISK_SYNTHESIS_USER_PROMPT_TEMPLATE = """Please review the following multi-source risk assessment data and provide a comprehensive Risk Memorandum:

### Quantitative Metrics:
- VaR (95% / 99%): {var_95:.2%} / {var_99:.2%}
- CVaR (95% / 99%): {cvar_95:.2%} / {cvar_99:.2%}
- Sharpe Ratio: {sharpe:.2f}
- Sortino Ratio: {sortino:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Annualized Return: {annualized_return:.2%}

### Monte Carlo Simulation ({mc_model}):
- Days Horizon: {mc_days}
- Simulated Paths: {mc_paths}
- Mean Terminal Return: {mc_mean_return:.2%}
- Median Return: {mc_median_return:.2%}
- Probability of Loss: {mc_loss_prob:.2%}

### Volatility & Time Series Forecast:
- Annualized Forecast Volatility: {forecast_vol:.2%}
- Expected Mean Cumulative Return: {forecast_return:.2%}

### Technical & Visual Market Pattern:
- Market Regime: {regime}
- Identified Key Patterns: {patterns}
- Dynamic Support: {support}
- Dynamic Resistance: {resistance}

Provide:
1. Executive Risk Summary & Risk Stance (Defensive / Neutral / Aggressive)
2. Tail Risk & Liquidity Exposure Analysis
3. Monte Carlo Scenario Breakdown
4. Actionable Risk Mitigation & Hedging Strategies (e.g. Put options, stop-loss triggers, portfolio rebalancing)
"""
