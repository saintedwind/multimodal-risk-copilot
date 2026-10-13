"""Numeric pattern heuristics and image integrity checks; no visual inference model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.time_series.preprocessor import prepare_prices


def rule_based_pattern_scanner(df: pd.DataFrame) -> Dict[str, Any]:
    """Scans for classical price action patterns and momentum divergences deterministically."""
    df = prepare_prices(df)
    closes = df["Close"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    n = len(closes)
    patterns = []
    regime = "Neutral"

    # Peak & Trough scanner over window
    if n >= 30:
        peak1_idx = int(np.argmax(highs[-30:-15])) + (n - 30)
        peak2_idx = int(np.argmax(highs[-15:])) + (n - 15)
        p1, p2 = highs[peak1_idx], highs[peak2_idx]

        if abs(p1 - p2) / p1 < 0.015 and p1 > np.mean(highs[-40:]):
            patterns.append(
                {
                    "pattern": "Double Top",
                    "bias": "Bearish",
                    "confidence": None,
                    "evidence_type": "unvalidated_rule",
                    "description": f"Resistance ceiling observed near {round(p1, 2)} with failed follow-through.",
                }
            )
            regime = "Bearish Reversal"

        # Check Head and Shoulders approximation
        if n >= 60:
            left = np.max(highs[-60:-40])
            head = np.max(highs[-40:-20])
            right = np.max(highs[-20:])
            if head > left and head > right and abs(left - right) / left < 0.03:
                patterns.append(
                    {
                        "pattern": "Head & Shoulders",
                        "bias": "Bearish",
                        "confidence": None,
                        "evidence_type": "unvalidated_rule",
                        "description": f"Head peak at {round(head, 2)} with symmetrical shoulders near {round(left, 2)}.",
                    }
                )
                regime = "Bearish Distribution"

    # Support / Resistance breakdown
    support = np.min(lows[-60:-10]) if n >= 60 else np.min(lows)
    current_close = closes[-1]
    if current_close < support * 0.995:
        patterns.append(
            {
                "pattern": "Support Breakdown",
                "bias": "Bearish",
                "confidence": None,
                "evidence_type": "unvalidated_rule",
                "description": f"Close {current_close:.2f} broken below dynamic support floor {support:.2f}.",
            }
        )
    elif current_close > (np.max(highs[-60:-10]) if n >= 60 else np.max(highs)):
        patterns.append(
            {
                "pattern": "Resistance Breakout",
                "bias": "Bullish",
                "confidence": None,
                "evidence_type": "unvalidated_rule",
                "description": "Breakout above structural horizontal resistance.",
            }
        )
        regime = "Bullish Expansion"

    if not patterns:
        patterns.append(
            {
                "pattern": "Mean Reverting Consolidation",
                "bias": "Neutral",
                "confidence": None,
                "evidence_type": "unvalidated_rule",
                "description": "Price oscillating within normal statistical bounds without terminal structural break.",
            }
        )

    return {
        "regime": regime,
        "patterns_detected": patterns,
        "key_support": float(np.min(lows[-40:]) if n >= 40 else np.min(lows)),
        "key_resistance": float(np.max(highs[-40:]) if n >= 40 else np.max(highs)),
        "multimodal_source": "RuleBasedTechnicalScanner",
    }


def analyze_chart(image_path: str, price_df=None):
    artifact = None
    if image_path:
        from PIL import Image

        p = Path(image_path)
        if p.stat().st_size > 5_000_000:
            raise ValueError("Image exceeds 5 MB")
        with Image.open(p) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("Image pixel budget exceeded")
            image.verify()
        artifact = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "status": "decoded_and_verified_only",
        }
    if price_df is None:
        return {
            "regime": None,
            "patterns_detected": [],
            "key_support": None,
            "key_resistance": None,
            "multimodal_source": "image_only_not_supported",
            "image_artifact": artifact,
        }
    result = rule_based_pattern_scanner(price_df)
    result["image_artifact"] = artifact
    return result
