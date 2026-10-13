"""OHLC evidence charts, EMA MACD and simple moving-average RSI."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((loss == 0) & (gain > 0), 100).mask(
        (loss == 0) & (gain == 0), 50
    )
    return result


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def render_chart(
    df: pd.DataFrame, output_path: str, title: str = "Market Technical Overview"
) -> str:
    """Renders a 3-panel professional technical chart (Candlestick, MACD, RSI)."""
    from src.time_series.preprocessor import prepare_prices

    df = prepare_prices(df)
    df["sma20"] = df.Close.rolling(20).mean()
    df["sma50"] = df.Close.rolling(50).mean()
    df["macd"], df["signal"], df["hist"] = compute_macd(df.Close)
    df["rsi"] = compute_rsi(df.Close)
    data = df.tail(120).copy().reset_index(drop=True)
    fig, (ax_main, ax_macd, ax_rsi) = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08},
    )

    indices = np.arange(len(data))
    width = 0.6
    width2 = 0.1

    up = data[data["Close"] >= data["Open"]]
    down = data[data["Close"] < data["Open"]]

    col_up = "#10b981"
    col_down = "#ef4444"

    # Candlestick bars
    ax_main.bar(
        up.index,
        up["Close"] - up["Open"],
        width,
        bottom=up["Open"],
        color=col_up,
        alpha=0.9,
    )
    ax_main.bar(
        up.index, up["High"] - up["Close"], width2, bottom=up["Close"], color=col_up
    )
    ax_main.bar(
        up.index, up["Low"] - up["Open"], width2, bottom=up["Open"], color=col_up
    )

    ax_main.bar(
        down.index,
        down["Close"] - down["Open"],
        width,
        bottom=down["Open"],
        color=col_down,
        alpha=0.9,
    )
    ax_main.bar(
        down.index,
        down["High"] - down["Open"],
        width2,
        bottom=down["Open"],
        color=col_down,
    )
    ax_main.bar(
        down.index,
        down["Low"] - down["Close"],
        width2,
        bottom=down["Close"],
        color=col_down,
    )

    # Moving averages
    ma20 = data["sma20"]
    ma50 = data["sma50"]
    ax_main.plot(indices, ma20, color="#3b82f6", linewidth=1.2, label="SMA 20")
    ax_main.plot(indices, ma50, color="#f59e0b", linewidth=1.2, label="SMA 50")
    ax_main.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_main.grid(True, linestyle="--", alpha=0.3)
    ax_main.legend(loc="upper left", framealpha=0.8)

    # MACD
    macd, sig, hist = (data["macd"], data["signal"], data["hist"])
    ax_macd.plot(indices, macd, color="#3b82f6", label="MACD", linewidth=1.0)
    ax_macd.plot(indices, sig, color="#f97316", label="Signal", linewidth=1.0)
    ax_macd.bar(
        indices,
        hist,
        color=np.where(hist >= 0, "#10b981", "#ef4444"),
        alpha=0.6,
        width=0.5,
    )
    ax_macd.set_ylabel("MACD", fontsize=9)
    ax_macd.grid(True, linestyle="--", alpha=0.3)
    ax_macd.legend(loc="upper left", fontsize=8)

    # RSI
    rsi = data["rsi"]
    ax_rsi.plot(indices, rsi, color="#8b5cf6", linewidth=1.2)
    ax_rsi.axhline(70, color="#ef4444", linestyle="--", alpha=0.6)
    ax_rsi.axhline(30, color="#10b981", linestyle="--", alpha=0.6)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI (14)", fontsize=9)
    ax_rsi.grid(True, linestyle="--", alpha=0.3)

    if "Date" in data:
        step = max(1, len(data) // 6)
        ticks = list(range(0, len(data), step))
        ax_rsi.set_xticks(ticks)
        ax_rsi.set_xticklabels(
            [str(data["Date"].iloc[i])[:10] for i in ticks], rotation=25
        )

    if isinstance(output_path, (str, os.PathLike)):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.94)
    fig.savefig(output_path, dpi=160, format="png")
    plt.close(fig)
    return output_path
