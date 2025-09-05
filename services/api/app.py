import os
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SkyBlock Econ Model API", version="0.1.0")

DATABASE_URL = os.getenv("DATABASE_URL")

class ForecastResponse(BaseModel):
    product_id: str
    horizon_minutes: int
    ts: str
    forecast_price: float
    model_version: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/forecast/{product_id}", response_model=ForecastResponse)
def get_forecast(product_id: str, horizon_minutes: int = 60):
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
              SELECT ts, forecast_price, model_version
              FROM model_forecasts
              WHERE product_id = %s AND horizon_minutes = %s
              ORDER BY ts DESC
              LIMIT 1
            """, (product_id, horizon_minutes))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Forecast not found")
            ts, price, version = row
            return ForecastResponse(
                product_id=product_id,
                horizon_minutes=horizon_minutes,
                ts=ts.isoformat(),
                forecast_price=float(price),
                model_version=version,
            )
    finally:
        conn.close()