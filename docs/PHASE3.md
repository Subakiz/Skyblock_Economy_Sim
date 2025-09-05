# Phase 3: Advanced ML and Agent-Based Modeling

This document outlines the Phase 3 implementation, which transforms the SkyBlock Economic Modeling system from a reactive analyst to a proactive predictive entity using advanced machine learning and agent-based modeling.

## Overview

Phase 3 introduces cutting-edge features that provide comprehensive market prediction and simulation capabilities:

- **Multivariate ML Models**: LightGBM and XGBoost models for sophisticated price prediction
- **Agent-Based Modeling**: Virtual player agents representing different SkyBlock player types
- **Market Simulation**: Realistic market dynamics with external shocks and player behavior
- **Predictive Analysis Engine**: Integration of ML and ABM for proactive market insights
- **Scenario Testing**: Comprehensive testing of market reactions to various events

## Architecture

```
Phase 3 Components:
├── ML Forecasting (modeling/forecast/ml_forecaster.py)
│   ├── LightGBM/XGBoost models
│   ├── Multivariate feature engineering
│   └── Advanced prediction capabilities
├── Agent-Based Modeling (modeling/agents/)
│   ├── Player agent archetypes
│   ├── Market simulation model
│   └── Behavioral economics
├── Simulation Engine (modeling/simulation/)
│   ├── Predictive market engine
│   ├── Scenario comparison
│   └── Market insight generation
└── API & CLI Interfaces
    ├── Enhanced API endpoints
    └── Command-line tools
```

## Components

### 1. Advanced ML Forecasting (`modeling/forecast/ml_forecaster.py`)

**Features:**
- **Multi-model Support**: Both LightGBM and XGBoost implementations
- **Multivariate Predictions**: Uses 16+ features including cross-item dependencies
- **Multiple Horizons**: Simultaneous predictions for 15min, 60min, and 240min
- **Feature Engineering**: Advanced features like momentum, MA ratios, market context
- **Time Series Validation**: Proper time series cross-validation for realistic metrics

**Key Features:**
```python
# Enhanced feature set
features = [
    'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
    'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
    'ma_crossover', 'ma_ratio', 'vol_price_ratio',
    'hour_of_day', 'day_of_week', 'day_of_month',
    'market_volatility', 'market_momentum'
]

# Market context variables
- Cross-item price correlations
- Market-wide volatility indicators  
- Temporal patterns and seasonality
- Volume-price relationships
```

**Usage:**
```bash
# Train LightGBM model
python -m modeling.forecast.ml_forecaster ENCHANTED_LAPIS_BLOCK lightgbm

# Train XGBoost model  
python -m modeling.forecast.ml_forecaster HYPERION xgboost
```

### 2. Agent-Based Modeling (`modeling/agents/`)

**Player Archetypes:**
- **Early Game Farmer**: Low-risk, basic item focus, limited capital
- **End Game Dungeon Runner**: High-value items, sophisticated strategies
- **Auction Flipper**: Rapid trading, market-making behavior
- **Whale Investor**: Patient, high-capital, long-term positions
- **Casual Player**: Infrequent activity, simple strategies

**Agent Properties:**
```python
@dataclass
class PlayerStats:
    skill_level: int        # 1-60 skill progression
    net_worth: float        # Total wealth
    daily_playtime: float   # Activity level
    risk_tolerance: float   # 0.0-1.0 risk appetite
    market_knowledge: float # 0.0-1.0 expertise level
    reaction_speed: float   # 0.0-1.0 decision speed
```

**Market Dynamics:**
- Supply/demand mechanics with price impact
- Market sentiment tracking
- External shock events (updates, mayor changes)
- Realistic transaction costs and slippage
- Inventory constraints and behavioral patterns

### 3. Market Simulation (`modeling/agents/market_simulation.py`)

**Simulation Features:**
- **Realistic Agent Population**: Diverse mix of player types (100+ agents)
- **Market Mechanics**: Supply/demand, sentiment, volatility modeling
- **External Events**: Game updates, mayor elections, dungeon events
- **Price Discovery**: Emergent pricing through agent interactions
- **Performance Tracking**: Agent profitability and strategy success

**Scenarios Available:**
- `normal_market`: Standard market conditions
- `volatile_market`: High volatility environment  
- `stable_market`: Low volatility, efficient market
- `major_update`: Simulates large game update impact

**Usage:**
```python
from modeling.agents.market_simulation import ScenarioEngine

engine = ScenarioEngine()
results = engine.run_scenario('normal_market', steps=1000)
comparison = engine.compare_scenarios(['normal_market', 'volatile_market'])
```

### 4. Predictive Analysis Engine (`modeling/simulation/predictive_engine.py`)

**Integration Features:**
- **ML + ABM Integration**: Uses ML predictions as initial conditions for simulations
- **Comprehensive Analysis**: Combines multiple data sources and methodologies
- **Trading Opportunities**: Identifies high-confidence trading signals
- **Risk Assessment**: Market risk analysis across scenarios
- **Accuracy Tracking**: Monitors ML prediction accuracy over time

**Key Capabilities:**
```python
class PredictiveMarketEngine:
    def train_predictive_models()     # Train ML models for items
    def generate_ml_predictions()     # Get current ML forecasts
    def simulate_market_scenarios()   # Run ABM simulations
    def generate_market_insights()    # Create actionable insights
    def run_full_analysis()          # Complete analysis pipeline
```

## API Endpoints

### ML Forecasting
```http
POST /ml/train
{
    "product_id": "ENCHANTED_LAPIS_BLOCK",
    "model_type": "lightgbm",
    "horizons": [15, 60, 240]
}
```

### Market Simulation
```http
POST /simulation/market
{
    "scenario": "normal_market",
    "n_agents": 100,
    "steps": 500,
    "market_volatility": 0.02
}
```

### Predictive Analysis
```http
POST /analysis/predictive
{
    "items": ["HYPERION", "NECRON_CHESTPLATE"],
    "model_type": "lightgbm",
    "scenarios": ["normal_market", "volatile_market"],
    "include_opportunities": true
}
```

### Scenario Management
```http
GET /scenarios/available           # List scenarios
POST /scenarios/compare           # Compare scenarios
GET /models/status               # Model status
```

## CLI Interface

The Phase 3 CLI provides comprehensive command-line access to all features:

```bash
# Check Phase 3 status
./phase3_cli.py status

# Train ML model
./phase3_cli.py train ENCHANTED_LAPIS_BLOCK --model-type lightgbm

# Run market simulation
./phase3_cli.py simulate --scenario normal_market --agents 100 --steps 1000

# Predictive analysis
./phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --output analysis.json

# Compare scenarios
./phase3_cli.py compare "normal_market,volatile_market" --output comparison.json

# List available scenarios
./phase3_cli.py scenarios
```

## Configuration

Phase 3 adds extensive configuration options in `config/config.yaml`:

```yaml
phase3:
  ml_models:
    default_model: "lightgbm"
    horizons: [15, 60, 240]
    retrain_interval_hours: 24
    feature_window_size: 500
    cross_validation_splits: 3
    
  agent_based_modeling:
    default_agent_count: 100
    simulation_steps: 1000
    market_volatility: 0.02
    enable_external_shocks: true
    
  scenario_testing:
    scenarios: [normal_market, volatile_market, stable_market, major_update]
    comparison_metrics: [price_volatility, agent_performance, transaction_volume]
      
  predictive_engine:
    cache_predictions_minutes: 30
    confidence_threshold: 0.6
    opportunity_threshold: 5.0
    max_opportunities: 10
```

## Testing & Validation

### Model Performance Validation

**ML Model Metrics:**
- Time series cross-validation with 3 splits
- Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
- Feature importance analysis
- Prediction confidence intervals

**Agent Behavior Validation:**
- Agent decision consistency
- Strategy profitability across scenarios
- Market efficiency measures
- Price discovery accuracy

### Scenario Testing

**Test Coverage:**
```bash
# Test ML forecasting
python -m pytest tests/test_ml_forecaster.py

# Test agent behavior
python -m pytest tests/test_agents.py

# Test market simulation
python -m pytest tests/test_market_simulation.py

# Test predictive engine
python -m pytest tests/test_predictive_engine.py
```

## Example Workflows

### 1. Complete Market Analysis

```python
from modeling.simulation.predictive_engine import PredictiveMarketEngine

engine = PredictiveMarketEngine()

# Analyze key items
results = engine.run_full_analysis([
    'ENCHANTED_LAPIS_BLOCK',
    'HYPERION',
    'NECRON_CHESTPLATE'
], model_type='lightgbm')

# Save comprehensive results
engine.save_analysis(results, 'market_analysis.json')
```

### 2. Trading Strategy Development

```bash
# Train models for target items
./phase3_cli.py train HYPERION --model-type lightgbm
./phase3_cli.py train NECRON_CHESTPLATE --model-type xgboost

# Test strategy across scenarios
./phase3_cli.py compare "normal_market,volatile_market,stable_market" --output strategy_test.json

# Generate trading signals
./phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --output signals.json
```

### 3. Market Research & Insights

```python
# Compare model performance
lightgbm_results = engine.run_scenario('normal_market', model_type='lightgbm')
xgboost_results = engine.run_scenario('normal_market', model_type='xgboost')

# Analyze market efficiency
efficiency_scores = engine.assess_market_efficiency(simulation_results)

# Generate insights
insights = engine.generate_market_insights(predictions, simulations)
```

## Performance Considerations

- **ML Training**: 500+ data points required per item for reliable models
- **Simulation Scale**: 100 agents × 1000 steps ≈ 30 seconds execution time
- **Memory Usage**: ~500MB for full analysis of 10 items
- **Prediction Caching**: 30-minute cache reduces API response time
- **Parallel Processing**: Multi-threaded simulation and training

## Known Limitations

1. **Data Requirements**: ML models require substantial historical data (500+ points)
2. **Computational Cost**: Full analysis can take 5-10 minutes for multiple items
3. **Market Impact**: Simulations don't account for external player reactions to predictions
4. **Model Drift**: ML models may need retraining as market conditions change
5. **Agent Simplification**: Real player behavior is more complex than modeled archetypes

## Future Enhancements

- **Deep Learning**: LSTM/Transformer models for sequence prediction
- **Real-time Adaptation**: Online learning and model updating
- **External Data**: Integration of external SkyBlock events and social media sentiment  
- **Portfolio Optimization**: Multi-item portfolio strategies
- **Risk Management**: Advanced risk metrics and position sizing

---

**Phase 3 Status**: ✅ Complete  
**Next Phase**: Production Deployment & Real-time Trading