import { apiClient } from './client';
import type {
  ForecastResponse,
  AHPriceResponse,
  CraftProfitResponse,
  BacktestRequest,
  BacktestResponse,
  MLForecastRequest,
  MLForecastResponse,
  MarketSimulationRequest,
  MarketSimulationResponse,
  PredictiveAnalysisRequest,
  PredictiveAnalysisResponse,
} from './types';

export const apiService = {
  // Health check
  healthCheck: () => apiClient.get('/healthz'),

  // Market data
  getForecast: (productId: string, horizonMinutes: number = 60) =>
    apiClient.get<ForecastResponse>(`/forecast/${productId}`, {
      params: { horizon_minutes: horizonMinutes },
    }),

  getAHPrices: (productId: string, window: string = '1h') =>
    apiClient.get<AHPriceResponse>(`/prices/ah/${productId}`, {
      params: { window },
    }),

  getCraftProfitability: (
    productId: string,
    horizon: string = '1h',
    pricing: string = 'median'
  ) =>
    apiClient.get<CraftProfitResponse>(`/profit/craft/${productId}`, {
      params: { horizon, pricing },
    }),

  // Backtesting
  runBacktest: (request: BacktestRequest) =>
    apiClient.post<BacktestResponse>('/backtest/run', request),

  // Phase 3: ML and Simulation
  trainMLModel: (request: MLForecastRequest) =>
    apiClient.post<MLForecastResponse>('/ml/train', request),

  runMarketSimulation: (request: MarketSimulationRequest) =>
    apiClient.post<MarketSimulationResponse>('/simulation/market', request),

  runPredictiveAnalysis: (request: PredictiveAnalysisRequest) =>
    apiClient.post<PredictiveAnalysisResponse>('/analysis/predictive', request),

  // Scenario management
  getAvailableScenarios: () =>
    apiClient.get<string[]>('/scenarios/available'),

  compareScenarios: (scenarioNames: string[], steps: number = 500) =>
    apiClient.post('/scenarios/compare', scenarioNames, { params: { steps } }),

  getModelStatus: () =>
    apiClient.get('/models/status'),
};