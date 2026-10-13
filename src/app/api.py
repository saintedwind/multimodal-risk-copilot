"""Bounded daily risk analysis API. No external paid calls."""

import io
from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.app.service import analyze_prices, analyze_returns
from src.data_sources import load_official

app = FastAPI(title="Market Risk Review Desk", version="0.3.0")
slots = BoundedSemaphore(2)


@contextmanager
def capacity():
    if not slots.acquire(blocking=False):
        raise HTTPException(503, "Analysis capacity busy; retry later")
    try:
        yield
    finally:
        slots.release()


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    returns: list[float] = Field(min_length=5, max_length=20000)
    risk_free_rate: float = Field(default=0.0, gt=-1)
    horizon_days: int = Field(default=60, ge=1, le=504)
    n_paths: int = Field(default=1000, ge=1, le=10000)
    model: Literal["gbm", "jump_diffusion"] = "gbm"
    initial_price: float = Field(default=100.0, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    jump_intensity: float = Field(default=0.0, ge=0, le=50)
    jump_mean: float = Field(default=0.0, ge=-1, le=1)
    jump_sigma: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def budget(self):
        if self.horizon_days * self.n_paths > 2_000_000:
            raise ValueError("Simulation budget exceeded")
        return self


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Market Risk Review Desk",
        "version": "0.3.0",
        "paid_api_enabled": False,
    }


@app.post("/analyze/metrics")
def metrics(req: RiskRequest):
    try:
        with capacity():
            return analyze_returns(**req.model_dump())
    except (ValueError, OverflowError) as e:
        raise HTTPException(400, str(e))


@app.get("/analyze/sample")
def sample(symbol: str = "SPY"):
    df = pd.read_csv(Path(__file__).resolve().parents[2] / "data/sample_prices.csv")
    try:
        with capacity():
            return analyze_prices(df[df.Symbol == symbol].reset_index(drop=True))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/analyze/full")
def full(
    file: UploadFile = File(...),
    price_col: str = Form("Close"),
    symbol: str | None = Form(None),
):
    raw = file.file.read(2_000_001)
    if len(raw) > 2_000_000:
        raise HTTPException(413, "CSV exceeds 2 MB")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=20001)
        if len(df) > 20000:
            raise ValueError("CSV exceeds 20000 rows")
        if symbol is not None:
            if "Symbol" not in df:
                raise ValueError("Symbol column is missing")
            df = df[df.Symbol == symbol].reset_index(drop=True)
        with capacity():
            return analyze_prices(df, price_col)
    except (ValueError, OverflowError, UnicodeError, pd.errors.ParserError) as e:
        raise HTTPException(400, str(e))


@app.get("/analyze/reference")
def reference(symbol: Literal["EURUSD", "GBPUSD"] = "EURUSD"):
    try:
        with capacity():
            df = load_official()
            return analyze_prices(df[df.Symbol == symbol].reset_index(drop=True))
    except ValueError as e:
        raise HTTPException(400, str(e))
