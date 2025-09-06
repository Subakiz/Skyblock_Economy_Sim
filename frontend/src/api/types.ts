// API Response Types based on the FastAPI Pydantic models

export interface ForecastResponse {
  product_id: string;
  horizon_minutes: number;
  ts: string;
  forecast_price: number;
  model_version: string;
}

export interface AHPriceResponse {
  product_id: string;
  window: string;
  median_price: number;
  p25_price: number;
  p75_price: number;
  sale_count: number;
  last_updated: string;
}

export interface CraftProfitResponse {
  product_id: string;
  craft_cost: number;
  expected_sale_price: number;
  gross_margin: number;
  net_margin: number;
  roi_percent: number;
  turnover_adj_profit: number;
  best_path: string;
  sell_volume: number;
  data_age_minutes: number;
}

export interface BacktestRequest {
  strategy: string;
  params: Record<string, any>;
  start_date: string;
  end_date: string;
  capital: number;
  item_id: string;
}

export interface BacktestResponse {
  total_return: number;
  total_return_pct: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  total_trades: number;
  win_rate: number;
  final_value: number;
}

export interface MLForecastRequest {
  product_id: string;
  model_type: 'lightgbm' | 'xgboost';
  horizons: number[];
}

export interface MLForecastResponse {
  product_id: string;
  model_type: string;
  predictions: Record<number, number>;
  model_metrics: Record<string, number>;
  training_completed: string;
}

export interface MarketSimulationRequest {
  scenario: string;
  n_agents: number;
  steps: number;
  market_volatility: number;
  initial_prices?: Record<string, number>;
}

export interface MarketSimulationResponse {
  scenario: string;
  results: Record<string, any>;
  agent_performance: Record<string, any>;
  price_changes: Record<string, number>;
  market_sentiment: number;
  total_trades: number;
}

export interface PredictiveAnalysisRequest {
  items: string[];
  model_type: 'lightgbm' | 'xgboost';
  scenarios: string[];
  include_opportunities: boolean;
}

export interface PredictiveAnalysisResponse {
  analysis_id: string;
  items: string[];
  price_forecasts: Record<string, Record<number, number>>;
  scenario_analysis: Record<string, any>;
  trading_opportunities: Array<{
    item: string;
    opportunity_type: string;
    confidence: number;
    expected_profit: number;
    risk_level: string;
  }>;
  market_outlook: Record<string, any>;
}

// Chart data types for visualization
export interface PriceDataPoint {
  timestamp: string;
  price: number;
  volume?: number;
}

export interface ChartSeries {
  name: string;
  data: PriceDataPoint[];
  color?: string;
}