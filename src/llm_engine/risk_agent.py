"""LLM Risk Agent: combines quantitative analytics, vision signals, and generates CRO-level intelligence."""
from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional
from src.llm_engine.prompts import RISK_EXECUTIVE_SYSTEM_PROMPT, RISK_SYNTHESIS_USER_PROMPT_TEMPLATE

class RiskCopilotAgent:
    """Intelligent agent orchestrating risk synthesis via LLM with fallback rule-based reasoning."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def synthesize_report(
        self,
        risk_metrics: Dict[str, Any],
        mc_summary: Dict[str, Any],
        ts_forecast: Dict[str, Any],
        vol_forecast: Dict[str, Any],
        pattern_data: Dict[str, Any],
    ) -> str:
        """Synthesize metrics into a cohesive narrative report using LLM or structured analytical generator."""
        patterns_str = ", ".join([p.get("pattern", "N/A") for p in pattern_data.get("patterns_detected", [])])
        
        user_prompt = RISK_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            var_95=risk_metrics.get("var_95", 0.0),
            var_99=risk_metrics.get("var_99", 0.0),
            cvar_95=risk_metrics.get("cvar_95", 0.0),
            cvar_99=risk_metrics.get("cvar_99", 0.0),
            sharpe=risk_metrics.get("sharpe", 0.0),
            sortino=risk_metrics.get("sortino", 0.0),
            max_drawdown=risk_metrics.get("max_drawdown", 0.0),
            annualized_return=risk_metrics.get("annualized_return", 0.0),
            mc_model=mc_summary.get("model", "gbm"),
            mc_days=mc_summary.get("days", 252),
            mc_paths=mc_summary.get("n_paths", 1000),
            mc_mean_return=mc_summary.get("mean_return", 0.0),
            mc_median_return=mc_summary.get("median_return", 0.0),
            mc_loss_prob=mc_summary.get("loss_probability", 0.0),
            forecast_vol=vol_forecast.get("annualized_volatility", 0.0),
            forecast_return=ts_forecast.get("cumulative_return", 0.0),
            regime=pattern_data.get("regime", "Neutral"),
            patterns=patterns_str or "No distinct pattern",
            support=f"{pattern_data.get('key_support', 0.0):.2f}",
            resistance=f"{pattern_data.get('key_resistance', 0.0):.2f}",
        )

        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RISK_EXECUTIVE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                )
                return str(response.choices[0].message.content)
            except Exception:
                pass

        # Robust, high-standard fallback report generator
        return self._generate_deterministic_report(
            risk_metrics, mc_summary, ts_forecast, vol_forecast, pattern_data
        )

    def _generate_deterministic_report(
        self,
        risk_metrics: Dict[str, Any],
        mc_summary: Dict[str, Any],
        ts_forecast: Dict[str, Any],
        vol_forecast: Dict[str, Any],
        pattern_data: Dict[str, Any],
    ) -> str:
        regime = pattern_data.get("regime", "Neutral")
        loss_prob = mc_summary.get("loss_probability", 0.0)
        var_95 = risk_metrics.get("var_95", 0.0)
        cvar_95 = risk_metrics.get("cvar_95", 0.0)
        sharpe = risk_metrics.get("sharpe", 0.0)
        mdd = risk_metrics.get("max_drawdown", 0.0)

        stance = "Defensive" if loss_prob > 0.45 or regime in ["Bearish Reversal", "Bearish Distribution"] else "Balanced"
        if sharpe > 1.2 and loss_prob < 0.35 and "Bullish" in regime:
            stance = "Constructive / Opportunistic"

        lines = [
            "# Multi-Modal Quantitative Risk Assessment Report",
            f"**Executive Risk Stance:** {stance}",
            "",
            "## 1. Executive Summary",
            f"- **Regime Diagnosis:** Current technical and structural environment is classified as **{regime}**.",
            f"- **Loss Probability:** 1-year forward Monte Carlo simulation estimates a **{loss_prob:.1%}** probability of negative return.",
            f"- **Tail Risk Expectation:** 95% Historical VaR is **{var_95:.2%}** with an expected tail loss (CVaR) of **{cvar_95:.2%}**.",
            f"- **Historical Risk-Adjusted Quality:** Sharpe Ratio is **{sharpe:.2f}** against a maximum drawdown of **{mdd:.2%}**.",
            "",
            "## 2. Quantitative Tail Risk & Volatility Profile",
            f"- **Value-at-Risk (99%):** {risk_metrics.get('var_99', 0.0):.2%}",
            f"- **Conditional VaR / Expected Shortfall (99%):** {risk_metrics.get('cvar_99', 0.0):.2%}",
            f"- **Forward Volatility:** Projected annualized volatility is **{vol_forecast.get('annualized_volatility', 0.0):.2%}** (EWMA/GARCH model).",
            f"- **Sortino Downside Quality:** {risk_metrics.get('sortino', 0.0):.2f}",
            "",
            "## 3. Forward Scenario Simulation (Monte Carlo)",
            f"- **Model:** {mc_summary.get('model', 'gbm').upper()} with {mc_summary.get('n_paths', 1000):,} paths over {mc_summary.get('days', 252)} trading days.",
            f"- **Mean Projected Return:** {mc_summary.get('mean_return', 0.0):.2%}",
            f"- **Median Projected Return:** {mc_summary.get('median_return', 0.0):.2%}",
            "",
            "## 4. Multi-Modal Vision & Technical Context",
            f"- **Structural Support:** {pattern_data.get('key_support', 0.0):.2f}",
            f"- **Structural Resistance:** {pattern_data.get('key_resistance', 0.0):.2f}",
            f"- **Identified Patterns:** " + ", ".join([p.get('pattern', '') for p in pattern_data.get('patterns_detected', [])]),
            "",
            "## 5. Recommended Mitigation & Hedging Mandates",
            "1. **Dynamic Stop-Loss Triggers:** Rebalance portfolio allocation if price crosses below structural support.",
            "2. **Tail-Risk Hedging:** Acquire out-of-the-money put spreads or increase defensive cash buffer if 99% CVaR threshold is exceeded.",
            "3. **Position Sizing:** Scale position size proportionally to inverse volatility to manage Sharpe drawdown limits."
        ]
        return "\n".join(lines)
