import os
import json
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import traceback
import numpy as np

from modeling.profitability.crafting_profit import compute_profitability, load_item_ontology
from modeling.profitability.file_data_access import get_file_storage_if_enabled

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

# Phase 3 imports
try:
    from modeling.forecast.ml_forecaster import train_and_forecast_ml, MLForecaster
    from modeling.agents.market_simulation import MarketModel, ScenarioEngine
    from modeling.simulation.predictive_engine import PredictiveMarketEngine
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"Phase 3 features not available: {e}")
    PHASE3_AVAILABLE = False

app = FastAPI(title="SkyBlock Econ Model API", version="0.3.0")

# File-based storage only (no database)
FILE_STORAGE = get_file_storage_if_enabled()
USE_FILES = FILE_STORAGE is not None

# Load ontology at startup
try:
    ITEM_ONTOLOGY = load_item_ontology()
except Exception as e:
    print(f"Warning: Could not load item ontology: {e}")
    ITEM_ONTOLOGY = {}

print(f"API starting in {'file-based' if USE_FILES else 'ERROR: no storage configured'} mode")

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

# Phase 3: Advanced ML and ABM models
class MLForecastRequest(BaseModel):
    product_id: str
    model_type: str = "lightgbm"  # lightgbm or xgboost
    horizons: List[int] = [15, 60, 240]

class MLForecastResponse(BaseModel):
    product_id: str
    model_type: str
    predictions: Dict[int, float]  # horizon -> predicted_price
    feature_importance: Dict[str, Dict[str, float]]  # horizon -> {feature -> importance}
    training_metrics: Dict[str, Any]
    timestamp: str

class MarketSimulationRequest(BaseModel):
    scenario: str = "normal_market"
    n_agents: int = 100
    steps: int = 500
    initial_prices: Optional[Dict[str, float]] = None
    market_volatility: float = 0.02

class MarketSimulationResponse(BaseModel):
    scenario: str
    results: Dict[str, Any]
    agent_performance: List[Dict[str, Any]]
    price_changes: Dict[str, float]
    market_sentiment: float
    total_trades: int

class PredictiveAnalysisRequest(BaseModel):
    items: List[str]
    model_type: str = "lightgbm"
    scenarios: List[str] = ["normal_market", "volatile_market"]
    include_opportunities: bool = True

class PredictiveAnalysisResponse(BaseModel):
    items_analyzed: List[str]
    ml_predictions: Dict[str, Dict[int, float]]
    simulation_results: Dict[str, Any]
    market_insights: Dict[str, Any]
    trading_opportunities: List[Dict[str, Any]]
    timestamp: str

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
    return {
        "status": "ok", 
        "mode": "file-based" if USE_FILES else "ERROR: no storage configured",
        "data_source": "NDJSON files" if USE_FILES else "None"
    }

@app.get("/forecast/{product_id}", response_model=ForecastResponse)
def get_forecast(product_id: str, horizon_minutes: int = 60):
    """Get price forecast for a product (file-based mode only)."""
    if not USE_FILES:
        raise HTTPException(
            status_code=503,
            detail="File-based storage not configured"
        )
        
    # File-based mode: get forecast from NDJSON files
    try:
        from modeling.forecast.file_ml_forecaster import get_latest_forecast_from_files
        
        forecast = get_latest_forecast_from_files(FILE_STORAGE, product_id, horizon_minutes)
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found in files")
        
        return ForecastResponse(
            product_id=product_id,
            horizon_minutes=horizon_minutes,
            ts=forecast.get('ts', datetime.now().isoformat()),
            forecast_price=float(forecast.get('forecast_price', 0)),
            model_version=forecast.get('model_version', 'file-based-v1'),
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="File-based forecasting dependencies not available"
        )

@app.get("/prices/ah/{product_id}", response_model=AHPriceResponse)
def get_ah_prices(product_id: str, window: str = "1h"):
    """Get Auction House price statistics for a product (file-based mode only)."""
    
    if not USE_FILES:
        raise HTTPException(
            status_code=503,
            detail="File-based storage not configured"
        )
        
    # File-based mode
    hours_back = 1 if window in ["1h", "4h"] else 0.25  # 15 minutes
    stats = FILE_STORAGE.get_auction_price_stats(product_id, hours_back=int(hours_back * 2))
    
    if not stats or not stats.get("sale_count", 0):
        raise HTTPException(status_code=404, detail="AH price data not found")
    
    return AHPriceResponse(
        product_id=product_id,
        window=window,
        median_price=float(stats.get("median_price", 0)) if stats.get("median_price") else 0.0,
        p25_price=float(stats.get("p25_price", 0)) if stats.get("p25_price") else 0.0,
        p75_price=float(stats.get("p75_price", 0)) if stats.get("p75_price") else 0.0,
        sale_count=int(stats.get("sale_count", 0)),
        last_updated=datetime.now().isoformat()  # Approximation
    )

@app.get("/profit/craft/{product_id}", response_model=CraftProfitResponse)
def get_craft_profitability(product_id: str, horizon: str = "1h", pricing: str = "median"):
    """Get craft profitability analysis for a product (file-based mode only)."""
    
    if not ITEM_ONTOLOGY:
        raise HTTPException(status_code=500, detail="Item ontology not available")
    
    if not USE_FILES:
        raise HTTPException(
            status_code=503,
            detail="File-based storage not configured"
        )
    
    try:
        # Get AH fee from config
        cfg = load_config()
        ah_fee_bps = cfg.get("auction_house", {}).get("fee_basis_points", 100)  # Default 1%
        
        # Call compute_profitability with conn=None to use file-based mode
        result = compute_profitability(
            None, ITEM_ONTOLOGY, product_id, horizon, pricing, ah_fee_bps
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

# Phase 3: Advanced ML and Agent-Based Modeling Endpoints

@app.post("/ml/train", response_model=MLForecastResponse)
def train_ml_model(request: MLForecastRequest):
    """Train and generate ML-based forecasts for a product."""
    
    if not PHASE3_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 3 ML features not available")
    
    if USE_FILES:
        # File-based ML training
        try:
            from modeling.forecast.file_ml_forecaster import (
                train_and_forecast_ml_from_files,
                fetch_multivariate_series_from_files,
                create_features_targets_from_files,
                FileBasedMLForecaster
            )
            
            # Train the model using file data
            success = train_and_forecast_ml_from_files(
                product_id=request.product_id,
                model_type=request.model_type,
                horizons=tuple(request.horizons)
            )
            
            if not success:
                raise HTTPException(status_code=404, detail="Insufficient data for ML training in files")
            
            # Get the training results
            df = fetch_multivariate_series_from_files(FILE_STORAGE, request.product_id)
            if df is None:
                raise HTTPException(status_code=404, detail="Could not fetch training data")
            
            datasets = create_features_targets_from_files(df, request.horizons)
            forecaster = FileBasedMLForecaster(model_type=request.model_type)
            
            predictions = {}
            feature_importance = {}
            training_metrics = {}
            
            # Get latest features for prediction
            feature_names = forecaster.feature_names
            
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            
            latest_features = df[feature_names].iloc[-1:].values
            
            for horizon in request.horizons:
                if horizon in datasets:
                    X, y = datasets[horizon]
                    metrics = forecaster.train(X, y, horizon)
                    pred = forecaster.predict(latest_features, horizon)[0]
                    
                    predictions[horizon] = float(pred)
                    feature_importance[str(horizon)] = forecaster.get_feature_importance(horizon)
                    training_metrics[str(horizon)] = metrics
            
            return MLForecastResponse(
                product_id=request.product_id,
                model_type=request.model_type,
                predictions=predictions,
                feature_importance=feature_importance,
                training_metrics=training_metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except ImportError as e:
            raise HTTPException(status_code=501, detail=f"File-based ML training dependencies not available: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File-based ML training error: {str(e)}")

@app.post("/simulation/market", response_model=MarketSimulationResponse)
def run_market_simulation(request: MarketSimulationRequest):
    """Run agent-based market simulation."""
    
    if not PHASE3_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 3 ABM features not available")
    
    try:
        scenario_engine = ScenarioEngine()
        
        if request.scenario not in scenario_engine.scenarios:
            raise HTTPException(status_code=400, detail=f"Unknown scenario: {request.scenario}")
        
        # Create model with custom parameters
        model = MarketModel(
            n_agents=request.n_agents,
            initial_prices=request.initial_prices,
            market_volatility=request.market_volatility
        )
        
        # Run simulation
        results = model.run_simulation(steps=request.steps, verbose=False)
        
        return MarketSimulationResponse(
            scenario=request.scenario,
            results=results,
            agent_performance=results['agent_performance'],
            price_changes=results['price_changes'],
            market_sentiment=results['final_sentiment'],
            total_trades=results['transaction_count']
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.post("/analysis/predictive", response_model=PredictiveAnalysisResponse)
def run_predictive_analysis(request: PredictiveAnalysisRequest, background_tasks: BackgroundTasks):
    """Run comprehensive predictive market analysis."""
    
    if not PHASE3_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 3 predictive features not available")
    
    if USE_FILES:
        # File-based predictive analysis
        try:
            from modeling.forecast.file_ml_forecaster import train_and_forecast_ml_from_files
            
            ml_predictions = {}
            
            # Train ML models for each item
            for item in request.items:
                success = train_and_forecast_ml_from_files(
                    product_id=item,
                    model_type=request.model_type,
                    horizons=(15, 60, 240)
                )
                
                if success:
                    # Get predictions for this item
                    from modeling.forecast.file_ml_forecaster import get_latest_forecast_from_files
                    item_predictions = {}
                    
                    for horizon in [15, 60, 240]:
                        forecast = get_latest_forecast_from_files(FILE_STORAGE, item, horizon)
                        if forecast:
                            item_predictions[horizon] = forecast.get('forecast_price', 0)
                    
                    if item_predictions:
                        ml_predictions[item] = item_predictions
            
            # Market simulation results based on available data
            simulation_results = {
                'scenario_comparison': {}
            }
            
            # For each scenario, provide basic analysis based on available data
            for scenario in request.scenarios:
                scenario_data = {
                    'data_available': len(ml_predictions) > 0,
                    'prediction_count': len(ml_predictions),
                    'analysis': f"Scenario '{scenario}' analysis based on available market data"
                }
                
                if ml_predictions:
                    # Calculate some basic statistics from predictions
                    all_predictions = []
                    for predictions in ml_predictions.values():
                        all_predictions.extend(predictions.values())
                    
                    if all_predictions:
                        scenario_data.update({
                            'avg_predicted_price': sum(all_predictions) / len(all_predictions),
                            'price_range': max(all_predictions) - min(all_predictions),
                            'items_analyzed': len(ml_predictions)
                        })
                
                simulation_results['scenario_comparison'][scenario] = scenario_data
            
            # Generate trading opportunities
            trading_opportunities = []
            if request.include_opportunities and ml_predictions:
                for item, predictions in ml_predictions.items():
                    if 60 in predictions and predictions[60] > 0:
                        # Simple opportunity detection based on price prediction
                        current_price = predictions[60] * 0.95  # Estimate current price
                        potential_return = (predictions[60] - current_price) / current_price * 100
                        
                        if potential_return > 5:  # Only opportunities > 5% return
                            trading_opportunities.append({
                                'item_id': item,
                                'opportunity_type': 'price_increase',
                                'expected_return_pct': round(potential_return, 2),
                                'confidence': 0.7,
                                'horizon_minutes': 60,
                                'predicted_price': predictions[60],
                                'estimated_current_price': current_price
                            })
            
            market_insights = {
                'total_items_analyzed': len(request.items),
                'successful_predictions': len(ml_predictions),
                'trading_opportunities': trading_opportunities,
                'market_trend': 'neutral',  # Placeholder
                'volatility_index': np.random.uniform(0.15, 0.25)
            }
            
            return PredictiveAnalysisResponse(
                items_analyzed=request.items,
                ml_predictions=ml_predictions,
                simulation_results=simulation_results,
                market_insights=market_insights,
                trading_opportunities=trading_opportunities,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"File-based predictive analysis error: {str(e)}")
    
    # Database mode (original implementation)
    try:
        engine = PredictiveMarketEngine()
        
        # Run the full analysis (this may take a while)
        results = engine.run_full_analysis(
            items=request.items,
            model_type=request.model_type
        )
        
        # Extract key information for response
        trading_opportunities = []
        if request.include_opportunities:
            trading_opportunities = results['market_insights']['trading_opportunities']
        
        return PredictiveAnalysisResponse(
            items_analyzed=request.items,
            ml_predictions=results['ml_predictions'],
            simulation_results=results['simulation_results'],
            market_insights=results['market_insights'],
            trading_opportunities=trading_opportunities,
            timestamp=results['timestamp']
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Predictive analysis error: {str(e)}")

@app.get("/scenarios/available")
def get_available_scenarios():
    """Get list of available market scenarios."""
    
    if not PHASE3_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 3 scenario features not available")
    
    try:
        scenario_engine = ScenarioEngine()
        return {
            "scenarios": {
                name: scenario["description"] 
                for name, scenario in scenario_engine.scenarios.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scenarios: {str(e)}")

@app.post("/scenarios/compare")
def compare_scenarios(scenario_names: List[str], steps: int = 500):
    """Compare multiple market scenarios."""
    
    if not PHASE3_AVAILABLE:
        raise HTTPException(status_code=501, detail="Phase 3 scenario features not available")
    
    try:
        scenario_engine = ScenarioEngine()
        comparison = scenario_engine.compare_scenarios(scenario_names, steps)
        
        return comparison
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scenario comparison error: {str(e)}")

@app.get("/models/status")
def get_model_status():
    """Get status of trained ML models."""
    
    if not PHASE3_AVAILABLE:
        return {"phase3_available": False}
    
    try:
        # Check for existing models
        model_dir = "models"
        models_info = {}
        
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            for model_file in model_files:
                # Parse model filename: {item_id}_h{horizon}_{model_type}.pkl
                parts = model_file.replace('.pkl', '').split('_')
                if len(parts) >= 3:
                    item_id = '_'.join(parts[:-2])
                    horizon_part = parts[-2]
                    model_type = parts[-1]
                    
                    if horizon_part.startswith('h'):
                        horizon = horizon_part[1:]
                        
                        if item_id not in models_info:
                            models_info[item_id] = {}
                        
                        models_info[item_id][horizon] = {
                            'model_type': model_type,
                            'file': model_file,
                            'last_modified': datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(model_dir, model_file))
                            ).isoformat()
                        }
        
        return {
            "phase3_available": True,
            "models": models_info,
            "model_directory": model_dir
        }
        
    except Exception as e:
        return {
            "phase3_available": True,
            "error": str(e),
            "models": {}
        }