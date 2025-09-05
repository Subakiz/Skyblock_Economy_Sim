"""
Phase 3: Integrated ML-ABM Scenario Engine
Combines machine learning predictions with agent-based modeling for proactive market analysis.
"""

import os
import yaml
import psycopg2
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from ..forecast.ml_forecaster import MLForecaster, fetch_multivariate_series, create_features_targets
from ..agents.market_simulation import MarketModel, ScenarioEngine

class PredictiveMarketEngine:
    """
    Advanced market analysis engine that combines ML predictions with ABM simulations
    to provide proactive market insights.
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or self._load_db_config()
        self.ml_models = {}  # item_id -> {horizon -> MLForecaster}
        self.scenario_engine = ScenarioEngine()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_timestamp = None
        
    def _load_db_config(self) -> Dict:
        """Load database configuration."""
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        return {
            'url': os.getenv("DATABASE_URL") or config["storage"]["database_url"]
        }
    
    def train_predictive_models(self, items: List[str], model_type: str = 'lightgbm') -> Dict[str, Any]:
        """Train ML models for specified items."""
        
        training_results = {}
        
        conn = psycopg2.connect(self.db_config['url'])
        
        try:
            for item_id in items:
                print(f"Training predictive model for {item_id}...")
                
                # Fetch data
                df = fetch_multivariate_series(conn, item_id)
                if df is None or df.empty:
                    print(f"Insufficient data for {item_id}")
                    continue
                
                # Create feature sets
                datasets = create_features_targets(df, [15, 60, 240])
                
                # Train models for each horizon
                item_models = {}
                for horizon, (X, y) in datasets.items():
                    forecaster = MLForecaster(model_type=model_type)
                    metrics = forecaster.train(X, y, horizon)
                    item_models[horizon] = forecaster
                    
                    print(f"{item_id} - Horizon {horizon}min: MAE={metrics['mae']:.2f}")
                
                self.ml_models[item_id] = item_models
                training_results[item_id] = {
                    'horizons_trained': list(item_models.keys()),
                    'data_points': len(df)
                }
        
        finally:
            conn.close()
        
        return training_results
    
    def generate_ml_predictions(self, items: List[str], horizons: List[int] = [15, 60, 240]) -> Dict[str, Dict[int, float]]:
        """Generate ML predictions for specified items and horizons."""
        
        predictions = {}
        
        conn = psycopg2.connect(self.db_config['url'])
        
        try:
            for item_id in items:
                if item_id not in self.ml_models:
                    print(f"No trained model for {item_id}")
                    continue
                
                # Get latest features
                df = fetch_multivariate_series(conn, item_id, min_points=50)
                if df is None or df.empty:
                    continue
                
                feature_names = [
                    'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
                    'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
                    'ma_crossover', 'ma_ratio', 'vol_price_ratio',
                    'hour_of_day', 'day_of_week', 'day_of_month',
                    'market_volatility', 'market_momentum'
                ]
                
                # Ensure all feature columns exist
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0.0
                
                latest_features = df[feature_names].iloc[-1:].values
                
                item_predictions = {}
                for horizon in horizons:
                    if horizon in self.ml_models[item_id]:
                        forecaster = self.ml_models[item_id][horizon]
                        pred = forecaster.predict(latest_features, horizon)[0]
                        item_predictions[horizon] = float(pred)
                
                predictions[item_id] = item_predictions
        
        finally:
            conn.close()
        
        return predictions
    
    def simulate_market_scenarios(self, 
                                 predictions: Dict[str, Dict[int, float]], 
                                 scenarios: List[str] = ['normal_market', 'volatile_market'],
                                 steps: int = 500) -> Dict[str, Any]:
        """
        Simulate market scenarios using ML predictions as initial conditions.
        """
        
        # Convert predictions to initial prices for simulation
        initial_prices = {}
        for item_id, item_preds in predictions.items():
            if 60 in item_preds:  # Use 60-minute predictions as initial prices
                initial_prices[item_id] = item_preds[60]
        
        scenario_results = {}
        
        for scenario in scenarios:
            print(f"Simulating scenario: {scenario}")
            
            # Create model with ML-predicted initial prices
            model = MarketModel(
                n_agents=100,
                initial_prices=initial_prices,
                **self.scenario_engine.scenarios[scenario]
            )
            
            # Run simulation
            results = model.run_simulation(steps, verbose=False)
            results['scenario'] = scenario
            results['initial_ml_predictions'] = predictions
            
            scenario_results[scenario] = results
        
        return scenario_results
    
    def analyze_prediction_accuracy(self, 
                                  historical_predictions: Dict[str, Any],
                                  actual_prices: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze the accuracy of ML predictions compared to actual market movements.
        """
        
        accuracy_metrics = {}
        
        for item_id in historical_predictions:
            if item_id not in actual_prices:
                continue
            
            item_metrics = {}
            
            for horizon, predicted_price in historical_predictions[item_id].items():
                actual = actual_prices[item_id]
                if len(actual) > horizon // 5:  # Assuming 5-minute intervals
                    actual_price = actual[horizon // 5]
                    
                    # Calculate accuracy metrics
                    absolute_error = abs(predicted_price - actual_price)
                    relative_error = absolute_error / actual_price * 100
                    
                    item_metrics[f"{horizon}min"] = {
                        'predicted': predicted_price,
                        'actual': actual_price,
                        'absolute_error': absolute_error,
                        'relative_error': relative_error
                    }
            
            accuracy_metrics[item_id] = item_metrics
        
        return accuracy_metrics
    
    def generate_market_insights(self, 
                               predictions: Dict[str, Dict[int, float]],
                               simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable market insights from ML predictions and simulations.
        """
        
        insights = {
            'price_forecasts': predictions,
            'scenario_analysis': {},
            'risk_assessment': {},
            'trading_opportunities': [],
            'market_outlook': {}
        }
        
        # Analyze simulation results
        for scenario, results in simulation_results.items():
            price_changes = results['price_changes']
            agent_performance = results['agent_performance']
            
            insights['scenario_analysis'][scenario] = {
                'expected_price_changes': price_changes,
                'agent_success_rates': self._analyze_agent_success(agent_performance),
                'market_volatility': results.get('final_volatility', 0),
                'sentiment': results.get('final_sentiment', 0)
            }
        
        # Identify trading opportunities
        opportunities = self._identify_opportunities(predictions, simulation_results)
        insights['trading_opportunities'] = opportunities
        
        # Risk assessment
        risk_metrics = self._assess_market_risks(simulation_results)
        insights['risk_assessment'] = risk_metrics
        
        # Market outlook
        outlook = self._generate_market_outlook(predictions, simulation_results)
        insights['market_outlook'] = outlook
        
        return insights
    
    def _analyze_agent_success(self, agent_performance: List[Dict]) -> Dict[str, float]:
        """Analyze success rates by agent archetype."""
        
        archetype_success = {}
        
        for agent in agent_performance:
            archetype = agent['archetype']
            if archetype not in archetype_success:
                archetype_success[archetype] = []
            
            # Consider positive performance as success
            success = agent['performance_pct'] > 0
            archetype_success[archetype].append(success)
        
        # Calculate success rates
        return {
            archetype: np.mean(successes) * 100
            for archetype, successes in archetype_success.items()
        }
    
    def _identify_opportunities(self, 
                              predictions: Dict[str, Dict[int, float]],
                              simulation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential trading opportunities."""
        
        opportunities = []
        
        # Look for items with strong predicted price movements
        for item_id, item_preds in predictions.items():
            if 60 in item_preds and 240 in item_preds:
                short_term = item_preds[60]
                long_term = item_preds[240]
                
                # Calculate expected return
                expected_return = (long_term - short_term) / short_term * 100
                
                if abs(expected_return) > 5:  # Significant movement expected
                    # Check simulation results for confidence
                    confidence = self._calculate_confidence(item_id, simulation_results)
                    
                    opportunity = {
                        'item_id': item_id,
                        'type': 'buy' if expected_return > 0 else 'sell',
                        'expected_return': expected_return,
                        'time_horizon': '240min',
                        'confidence': confidence,
                        'current_price': short_term,
                        'target_price': long_term
                    }
                    
                    opportunities.append(opportunity)
        
        # Sort by expected return and confidence
        opportunities.sort(key=lambda x: abs(x['expected_return']) * x['confidence'], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
    
    def _calculate_confidence(self, item_id: str, simulation_results: Dict[str, Any]) -> float:
        """Calculate confidence in prediction based on simulation consistency."""
        
        price_changes = []
        
        for scenario, results in simulation_results.items():
            if item_id in results['price_changes']:
                price_changes.append(results['price_changes'][item_id])
        
        if not price_changes:
            return 0.5  # Default confidence
        
        # Confidence based on consistency across scenarios
        consistency = 1.0 - (np.std(price_changes) / (np.mean(np.abs(price_changes)) + 0.01))
        return max(0.1, min(1.0, consistency))
    
    def _assess_market_risks(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall market risks."""
        
        risks = {
            'volatility_risk': 'low',
            'liquidity_risk': 'low', 
            'systemic_risk': 'low',
            'scenario_divergence': 0.0
        }
        
        # Calculate volatility across scenarios
        all_volatilities = []
        all_sentiments = []
        
        for scenario, results in simulation_results.items():
            price_changes = list(results['price_changes'].values())
            volatility = np.std(price_changes)
            all_volatilities.append(volatility)
            all_sentiments.append(results.get('final_sentiment', 0))
        
        avg_volatility = np.mean(all_volatilities)
        sentiment_range = max(all_sentiments) - min(all_sentiments)
        
        # Risk classification
        if avg_volatility > 20:
            risks['volatility_risk'] = 'high'
        elif avg_volatility > 10:
            risks['volatility_risk'] = 'medium'
        
        if sentiment_range > 1.0:
            risks['systemic_risk'] = 'high'
        elif sentiment_range > 0.5:
            risks['systemic_risk'] = 'medium'
        
        risks['scenario_divergence'] = sentiment_range
        
        return risks
    
    def _generate_market_outlook(self, 
                               predictions: Dict[str, Dict[int, float]],
                               simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market outlook."""
        
        # Aggregate predictions
        short_term_changes = []
        long_term_changes = []
        
        for item_id, preds in predictions.items():
            if 60 in preds and 240 in preds:
                # Assume current price is the 15-minute prediction
                current_price = preds.get(15, preds[60])
                
                short_change = (preds[60] - current_price) / current_price * 100
                long_change = (preds[240] - current_price) / current_price * 100
                
                short_term_changes.append(short_change)
                long_term_changes.append(long_change)
        
        outlook = {
            'short_term_trend': 'bullish' if np.mean(short_term_changes) > 1 else 'bearish' if np.mean(short_term_changes) < -1 else 'neutral',
            'long_term_trend': 'bullish' if np.mean(long_term_changes) > 2 else 'bearish' if np.mean(long_term_changes) < -2 else 'neutral',
            'expected_volatility': np.mean([np.std(list(results['price_changes'].values())) 
                                          for results in simulation_results.values()]),
            'market_efficiency': self._assess_market_efficiency(simulation_results)
        }
        
        return outlook
    
    def _assess_market_efficiency(self, simulation_results: Dict[str, Any]) -> float:
        """Assess market efficiency based on agent performance variation."""
        
        all_performances = []
        
        for scenario, results in simulation_results.items():
            performances = [agent['performance_pct'] for agent in results['agent_performance']]
            all_performances.extend(performances)
        
        # Lower variance in performance = more efficient market
        performance_variance = np.var(all_performances)
        efficiency = 1.0 / (1.0 + performance_variance / 100)  # Normalize to 0-1
        
        return efficiency
    
    def run_full_analysis(self, items: List[str], model_type: str = 'lightgbm') -> Dict[str, Any]:
        """
        Run complete predictive market analysis including training, prediction, 
        simulation, and insight generation.
        """
        
        print("Starting comprehensive market analysis...")
        
        # Step 1: Train ML models
        print("Step 1: Training ML models...")
        training_results = self.train_predictive_models(items, model_type)
        
        # Step 2: Generate predictions
        print("Step 2: Generating ML predictions...")
        predictions = self.generate_ml_predictions(items)
        
        # Step 3: Run simulations
        print("Step 3: Running market simulations...")
        simulation_results = self.simulate_market_scenarios(
            predictions, 
            ['normal_market', 'volatile_market', 'stable_market']
        )
        
        # Step 4: Generate insights
        print("Step 4: Generating market insights...")
        insights = self.generate_market_insights(predictions, simulation_results)
        
        # Combine all results
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'items_analyzed': items,
            'model_type': model_type,
            'training_results': training_results,
            'ml_predictions': predictions,
            'simulation_results': simulation_results,
            'market_insights': insights
        }
        
        print("Analysis complete!")
        return full_results
    
    def save_analysis(self, results: Dict[str, Any], filepath: str):
        """Save analysis results to file."""
        
        # Convert any numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f"Analysis saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    engine = PredictiveMarketEngine()
    
    # Analyze key SkyBlock items
    key_items = [
        'ENCHANTED_LAPIS_BLOCK',
        'HYPERION', 
        'NECRON_CHESTPLATE',
        'WITHER_SKULL'
    ]
    
    results = engine.run_full_analysis(key_items, model_type='lightgbm')
    engine.save_analysis(results, 'phase3_market_analysis.json')