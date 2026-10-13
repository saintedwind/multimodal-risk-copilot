"""Immutable numeric evidence; optional loopback model selects observations only."""

import json
import os
from urllib.request import ProxyHandler, Request, build_opener


class RiskCopilotAgent:
    def __init__(self, api_key=None, model=None, provider=None):
        self.provider = provider or os.getenv("REPORT_PROVIDER", "deterministic")
        self.last_status = "deterministic"
        if self.provider not in {"deterministic", "local"}:
            raise ValueError("Provider must be deterministic or local")

    def synthesize_report(
        self, risk_metrics, mc_summary, ts_forecast, vol_forecast, pattern_data
    ):
        fmt = lambda v: f"{v:.2%}" if v is not None else "undefined"
        lines = [
            f"Daily historical loss VaR 95%: {fmt(risk_metrics['var_95'])}; conditional tail mean: {fmt(risk_metrics['cvar_95'])}.",
            f"Historical maximum drawdown: {fmt(risk_metrics['max_drawdown'])}.",
            f"{mc_summary['model']} scenario loss probability over {mc_summary['days']} trading days: {fmt(mc_summary['loss_probability'])}; {mc_summary['n_paths']} paths; seed {mc_summary['seed']}.",
            f"EWMA annualized volatility: {fmt(vol_forecast['annualized_volatility'])}.",
            f"{ts_forecast['method']} baseline cumulative return over {ts_forecast['horizon_days']} days: {fmt(ts_forecast['cumulative_return'])}.",
            "Technical patterns are unvalidated numeric rules, not a visual-model diagnosis.",
        ]
        selected = list(range(len(lines)))
        self.last_status = "deterministic"
        if self.provider == "local":
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        }
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                }
                payload = {
                    "model": "local",
                    "temperature": 0,
                    "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "observations",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": "Select up to three distinct numbered observations most useful for a risk review. Output only JSON steps. Do not create advice. /no_think",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(dict(enumerate(lines, 1))),
                        },
                    ],
                }
                req = Request(
                    "http://127.0.0.1:8081/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with build_opener(ProxyHandler({})).open(req, timeout=30) as r:
                    data = json.loads(r.read(32768))
                c = data["choices"][0]
                v = json.loads(c["message"]["content"])
                steps = v["steps"]
                if (
                    c["finish_reason"] != "stop"
                    or set(v) != {"steps"}
                    or not 1 <= len(steps) <= 3
                    or any(
                        type(i) is not int or not 1 <= i <= len(lines) for i in steps
                    )
                    or len(set(steps)) != len(steps)
                ):
                    raise ValueError("Invalid selection")
                selected = [i - 1 for i in steps]
                self.last_status = "local_selection"
                self.last_usage = data.get("usage", {})
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                self.last_status = "local_unavailable_deterministic_fallback"
        return (
            "# Risk review notes\n\n"
            + "\n".join("- " + lines[i] for i in selected)
            + "\n\nDescriptive model outputs; not investment instructions. Scenario probabilities are conditional on fitted assumptions. Data provenance is not independently verified."
        )
