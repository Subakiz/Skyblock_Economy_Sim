import os
import json
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

from modeling.profitability.crafting_profit import compute_profitability, load_item_ontology

app = FastAPI(title="SkyBlock Econ Model API", version="0.2.0")

DATABASE_URL = os.getenv("DATABASE_URL")

# Load ontology at startup
try:
    ITEM_ONTOLOGY = load_item_ontology()
except Exception as e:
    print(f"Warning: Could not load item ontology: {e}")
    ITEM_ONTOLOGY = {}

class ForecastResponse(BaseModel):
    product_id: str
    horizon_minutes: int
    ts: str
    forecast_price: float
    model_version: str

class AHPriceResponse(BaseModel):
    product_id: str
    window: str
    median_price: float
    p25_price: float
    p75_price: float
    sale_count: int
    last_updated: str

class CraftProfitResponse(BaseModel):
    product_id: str
    craft_cost: float
    expected_sale_price: float
    gross_margin: float
    net_margin: float
    roi_percent: float
    turnover_adj_profit: float
    best_path: str
    sell_volume: int
    data_age_minutes: int

class BacktestRequest(BaseModel):
    strategy: str
    params: Dict[str, Any]
    start_date: str
    end_date: str
    capital: float
    item_id: Optional[str] = None

class BacktestResponse(BaseModel):
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    final_value: float

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

@app.get("/prices/ah/{product_id}", response_model=AHPriceResponse)
def get_ah_prices(product_id: str, window: str = "1h"):
    """Get Auction House price statistics for a product."""
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    
    # Map window to view table
    view_name = "ah_prices_1h" if window in ["1h", "4h"] else "ah_prices_15m"
    
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT median_price, p25_price, p75_price, sale_count, time_bucket
                FROM {view_name}
                WHERE item_id = %s
                ORDER BY time_bucket DESC
                LIMIT 1
            """, (product_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="AH price data not found")
                
            median, p25, p75, count, last_updated = row
            
            return AHPriceResponse(
                product_id=product_id,
                window=window,
                median_price=float(median) if median else 0.0,
                p25_price=float(p25) if p25 else 0.0,
                p75_price=float(p75) if p75 else 0.0,
                sale_count=int(count) if count else 0,
                last_updated=last_updated.isoformat()
            )
    finally:
        conn.close()

@app.get("/profit/craft/{product_id}", response_model=CraftProfitResponse)
def get_craft_profitability(product_id: str, horizon: str = "1h", pricing: str = "median"):
    """Get craft profitability analysis for a product."""
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    
    if not ITEM_ONTOLOGY:
        raise HTTPException(status_code=500, detail="Item ontology not available")
    
    conn = psycopg2.connect(DATABASE_URL)
    try:
        # Get AH fee from config (hardcoded for now)
        ah_fee_bps = 100
        
        result = compute_profitability(
            conn, ITEM_ONTOLOGY, product_id, horizon, pricing, ah_fee_bps
        )
        
        return CraftProfitResponse(
            product_id=result.product_id,
            craft_cost=result.craft_cost,
            expected_sale_price=result.expected_sale_price,
            gross_margin=result.gross_margin,
            net_margin=result.net_margin,
            roi_percent=result.roi_percent,
            turnover_adj_profit=result.turnover_adj_profit,
            best_path=result.best_path,
            sell_volume=result.sell_volume,
            data_age_minutes=result.data_age_minutes
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        conn.close()

@app.post("/backtest/run", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """Run a backtest simulation."""
    # This is a simplified implementation for demo purposes
    # In practice, you'd run the full backtesting engine
    
    try:
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        duration_days = (end_date - start_date).days
        
        # Simple mock response (would be replaced with actual backtesting)
        mock_return = request.capital * 0.15  # 15% mock return
        
        return BacktestResponse(
            total_return=mock_return,
            total_return_pct=15.0,
            max_drawdown=request.capital * 0.05,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.2,
            total_trades=25,
            win_rate=68.0,
            final_value=request.capital + mock_return
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Date parsing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {e}")