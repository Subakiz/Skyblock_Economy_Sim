"""
File-Based Predictive Market Engine

File-based version of PredictiveMarketEngine that works with NDJSON files
instead of database connections. Supports both Bazaar and Auction House data.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from storage.ndjson_storage import get_storage_instance
from modeling.features.auction_feature_pipeline import (
    build_auction_features_from_files,
    load_auction_features_from_file
)
from modeling.forecast.file_ml_forecaster import FileBasedMLForecaster
from modeling.agents.market_simulation import MarketModel, ScenarioEngine


class FileBasedPredictiveMarketEngine:
    """
    File-based market analysis engine that combines ML predictions with ABM simulations
    using NDJSON data files instead of database connections.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.ml_models = {}  # item_id -> {horizon -> FileBasedMLForecaster}
        self.scenario_engine = ScenarioEngine()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_timestamp = None
        
        # Storage instance
        self.storage = get_storage_instance(self.config)
        if self.storage is None:
            raise ValueError("No-database mode not enabled. Check config.yaml")
        
    def _load_config(self) -> Dict:
        """Load configuration."""
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    
    def prepare_auction_data(self, items: List[str]) -> bool:
        """Prepare auction features if they don't exist."""
        
        data_directory = self.config.get("no_database_mode", {}).get("data_directory", "data")
        features_path = Path(data_directory) / "auction_features.ndjson"
        
        if not features_path.exists():
            print("Auction features not found. Building from raw auction data...")
            try:
                build_auction_features_from_files()
                return features_path.exists()
            except Exception as e:
                print(f"Failed to build auction features: {e}")
                return False
        
        return True
    
    def load_auction_data_for_modeling(self, items: List[str]) -> Optional[pd.DataFrame]:
        """Load auction data prepared for ML modeling."""
        
        data_directory = self.config.get("no_database_mode", {}).get("data_directory", "data")
        
        # Load auction features
        df = load_auction_features_from_file(data_directory, items)
        
        if df is None or df.empty:
            return None
        
        # Prepare features for ML
        # Convert categorical features to numeric
        df['is_clean_numeric'] = df.get('is_clean', False).astype(int)
        df['has_reforge_numeric'] = df.get('has_reforge', False).astype(int)
        df['bin_numeric'] = df.get('bin', False).astype(int)
        
        # Create time-based features
        if 'ts' in df.columns:
            df['hour_of_day'] = df['ts'].dt.hour
            df['day_of_week'] = df['ts'].dt.dayofweek
            df['day_of_month'] = df['ts'].dt.day
        
        # Use final_price as the main price column for modeling
        if 'final_price' in df.columns:
            df['price'] = df['final_price']
        
        # Create price-based features
        if 'price' in df.columns:
            # Group by item_id to calculate item-specific features
            def add_price_features(group):
                group = group.sort_values('ts') if 'ts' in group.columns else group
                
                # Price lags
                group['price_lag_1'] = group['price'].shift(1)
                group['price_lag_5'] = group['price'].shift(5)
                
                # Moving averages
                group['ma_15'] = group['price'].rolling(window=15, min_periods=1).mean()
                group['ma_60'] = group['price'].rolling(window=60, min_periods=1).mean()
                
                # Momentum
                group['momentum_1'] = group['price'].pct_change(1).fillna(0)
                group['momentum_5'] = group['price'].pct_change(5).fillna(0)
                
                # Volatility
                group['vol_window_30'] = group['price'].rolling(window=30, min_periods=1).std()
                
                return group
            
            if 'item_id' in df.columns:
                df = df.groupby('item_id').apply(add_price_features).reset_index(drop=True)
        
        # Fill any remaining NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def create_features_targets_for_auction_data(self, df: pd.DataFrame, horizons: List[int] = [15, 60, 240]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Create feature matrices and target vectors for auction data."""
        
        feature_cols = [
            'is_clean_numeric', 'has_reforge_numeric', 'bin_numeric',
            'enchantment_count', 'price_per_hour',
            'ma_15', 'ma_60', 'vol_window_30',
            'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
            'hour_of_day', 'day_of_week', 'day_of_month',
            'rolling_avg_price_24h', 'rolling_avg_price_7d'
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Convert to numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = df[feature_cols].values
        
        datasets = {}
        for horizon in horizons:
            # For auction data, we create future price targets differently
            # Use a simple forward shift based on time
            if 'price' in df.columns:
                # Sort by timestamp
                if 'ts' in df.columns:
                    df_sorted = df.sort_values('ts')
                else:
                    df_sorted = df
                
                # Create target by shifting prices forward
                horizon_steps = max(1, horizon // 60)  # Assuming hourly intervals for auctions
                y = df_sorted['price'].shift(-horizon_steps).values
                
                # Use the sorted order for features too
                X_sorted = df_sorted[feature_cols].values
                
                # Remove rows where target is NaN
                valid_indices = ~np.isnan(y)
                X_valid = X_sorted[valid_indices]
                y_valid = y[valid_indices]
                
                if len(X_valid) > 50:  # Minimum data requirement for auctions
                    datasets[horizon] = (X_valid, y_valid)
        
        return datasets
    
    def train_predictive_models_from_files(self, items: List[str], model_type: str = 'lightgbm') -> Dict[str, Any]:
        """Train ML models for specified items using file data."""
        
        training_results = {}
        
        # Prepare auction data
        if not self.prepare_auction_data(items):
            print("Failed to prepare auction data")
            return training_results
        
        # Load auction data for modeling
        df = self.load_auction_data_for_modeling(items)
        
        if df is None or df.empty:
            print("No auction data available for modeling")
            return training_results
        
        for item_id in items:
            print(f"Training predictive model for {item_id}...")
            
            # Filter data for this item
            item_df = df[df.get('item_id') == item_id] if 'item_id' in df.columns else df
            
            if item_df.empty or len(item_df) < 100:
                print(f"Insufficient data for {item_id} ({len(item_df)} records)")
                continue
            
            # Create feature sets
            datasets = self.create_features_targets_for_auction_data(item_df, [15, 60, 240])
            
            if not datasets:
                print(f"Could not create datasets for {item_id}")
                continue
            
            # Train models for each horizon
            item_models = {}
            for horizon, (X, y) in datasets.items():
                forecaster = FileBasedMLForecaster(model_type=model_type)
                metrics = forecaster.train(X, y, horizon)
                item_models[horizon] = forecaster
                
                print(f"{item_id} - Horizon {horizon}min: MAE={metrics['mae']:.2f}")
            
            self.ml_models[item_id] = item_models
            training_results[item_id] = {
                'horizons_trained': list(item_models.keys()),
                'data_points': len(item_df)
            }
            
            # Save models
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            for horizon, forecaster in item_models.items():
                forecaster.save_model(f"{model_dir}/{item_id}", horizon)
        
        return training_results
    
    def generate_ml_predictions_from_files(self, items: List[str], horizons: List[int] = [15, 60, 240]) -> Dict[str, Dict[int, float]]:
        """Generate ML predictions from file data."""
        
        predictions = {}
        
        # Load auction data
        df = self.load_auction_data_for_modeling(items)
        if df is None or df.empty:
            print("No data available for predictions")
            return predictions
        
        feature_cols = [
            'is_clean_numeric', 'has_reforge_numeric', 'bin_numeric',
            'enchantment_count', 'price_per_hour',
            'ma_15', 'ma_60', 'vol_window_30',
            'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
            'hour_of_day', 'day_of_week', 'day_of_month',
            'rolling_avg_price_24h', 'rolling_avg_price_7d'
        ]
        
        for item_id in items:
            if item_id not in self.ml_models:
                print(f"No trained model for {item_id}")
                continue
            
            # Get latest features for this item
            item_df = df[df.get('item_id') == item_id] if 'item_id' in df.columns else df
            
            if item_df.empty:
                continue
            
            # Get the most recent record
            if 'ts' in item_df.columns:
                latest_record = item_df.sort_values('ts').iloc[-1:]
            else:
                latest_record = item_df.iloc[-1:]
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in latest_record.columns:
                    latest_record[col] = 0.0
            
            latest_features = latest_record[feature_cols].values
            
            item_predictions = {}
            for horizon in horizons:
                if horizon in self.ml_models[item_id]:
                    try:
                        forecaster = self.ml_models[item_id][horizon]
                        pred = forecaster.predict(latest_features, horizon)[0]
                        item_predictions[horizon] = float(pred)
                    except Exception as e:
                        print(f"Error predicting {item_id} horizon {horizon}: {e}")
            
            if item_predictions:
                predictions[item_id] = item_predictions
        
        return predictions
    
    def simulate_market_scenarios(self, 
                                 predictions: Dict[str, Dict[int, float]], 
                                 scenarios: List[str] = ['normal_market', 'volatile_market'],
                                 steps: int = 500) -> Dict[str, Any]:
        """Simulate market scenarios using ML predictions as initial conditions."""
        
        # Convert predictions to initial prices for simulation
        initial_prices = {}
        for item_id, item_preds in predictions.items():
            if 60 in item_preds:  # Use 60-minute predictions as initial prices
                initial_prices[item_id] = item_preds[60]
        
        scenario_results = {}
        
        for scenario in scenarios:
            print(f"Simulating scenario: {scenario}")
            
            try:
                # Get scenario configuration and filter for MarketModel parameters
                scenario_config = self.scenario_engine.scenarios.get(scenario, {})
                
                # Extract MarketModel parameters
                model_params = {}
                n_agents = scenario_config.get('n_agents', 100)
                if 'volatility' in scenario_config:
                    model_params['market_volatility'] = scenario_config['volatility'] 
                if 'shocks' in scenario_config:
                    model_params['external_shocks'] = scenario_config['shocks']
                
                # Create model with ML-predicted initial prices
                model = MarketModel(
                    n_agents=n_agents,
                    initial_prices=initial_prices,
                    **model_params
                )
                
                # Run simulation
                results = model.run_simulation(steps, verbose=False)
                results['scenario'] = scenario
                results['initial_ml_predictions'] = predictions
                
                scenario_results[scenario] = results
                
            except Exception as e:
                print(f"Error simulating scenario {scenario}: {e}")
                # Create minimal fallback result
                scenario_results[scenario] = {
                    'scenario': scenario,
                    'price_changes': {item_id: 0.0 for item_id in initial_prices.keys()},
                    'agent_performance': [],
                    'final_sentiment': 0.0,
                    'transaction_count': 0,
                    'final_volatility': 0.0,
                    'error': str(e)
                }
        
        return scenario_results
    
    def generate_market_insights(self, 
                               predictions: Dict[str, Dict[int, float]],
                               simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable market insights from ML predictions and simulations."""
        
        insights = {
            'price_forecasts': predictions,
            'scenario_analysis': {},
            'risk_assessment': {},
            'trading_opportunities': [],
            'market_outlook': {}
        }
        
        # Analyze simulation results
        for scenario, results in simulation_results.items():
            price_changes = results.get('price_changes', {})
            agent_performance = results.get('agent_performance', [])
            
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
        
        if not agent_performance:
            return {}
        
        archetype_success = {}
        
        for agent in agent_performance:
            archetype = agent.get('archetype', 'unknown')
            if archetype not in archetype_success:
                archetype_success[archetype] = []
            
            # Consider positive performance as success
            success = agent.get('performance_pct', 0) > 0
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
                expected_return = (long_term - short_term) / short_term * 100 if short_term > 0 else 0
                
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
            if item_id in results.get('price_changes', {}):
                price_changes.append(results['price_changes'][item_id])
        
        if not price_changes:
            return 0.5  # Default confidence
        
        # Confidence based on consistency across scenarios
        if len(price_changes) > 1:
            mean_abs = np.mean(np.abs(price_changes))
            consistency = 1.0 - (np.std(price_changes) / (mean_abs + 0.01))
            return max(0.1, min(1.0, consistency))
        else:
            return 0.7  # Single scenario confidence
    
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
            price_changes = list(results.get('price_changes', {}).values())
            if price_changes:
                volatility = np.std(price_changes)
                all_volatilities.append(volatility)
            all_sentiments.append(results.get('final_sentiment', 0))
        
        if all_volatilities:
            avg_volatility = np.mean(all_volatilities)
            sentiment_range = max(all_sentiments) - min(all_sentiments) if all_sentiments else 0
            
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
                # Use 60-minute as current price baseline
                current_price = preds[60]
                
                if current_price > 0:
                    short_change = 0  # 60min is our baseline
                    long_change = (preds[240] - current_price) / current_price * 100
                    
                    short_term_changes.append(short_change)
                    long_term_changes.append(long_change)
        
        # Calculate expected volatility from simulations
        volatilities = []
        for results in simulation_results.values():
            price_changes = list(results.get('price_changes', {}).values())
            if price_changes:
                volatilities.append(np.std(price_changes))
        
        outlook = {
            'short_term_trend': 'neutral',  # 60min baseline
            'long_term_trend': 'bullish' if np.mean(long_term_changes) > 2 else 'bearish' if np.mean(long_term_changes) < -2 else 'neutral' if long_term_changes else 'neutral',
            'expected_volatility': np.mean(volatilities) if volatilities else 0.0,
            'market_efficiency': self._assess_market_efficiency(simulation_results)
        }
        
        return outlook
    
    def _assess_market_efficiency(self, simulation_results: Dict[str, Any]) -> float:
        """Assess market efficiency based on agent performance variation."""
        
        all_performances = []
        
        for scenario, results in simulation_results.items():
            performances = [agent.get('performance_pct', 0) for agent in results.get('agent_performance', [])]
            all_performances.extend(performances)
        
        if not all_performances:
            return 0.5  # Default efficiency
        
        # Lower variance in performance = more efficient market
        performance_variance = np.var(all_performances)
        efficiency = 1.0 / (1.0 + performance_variance / 100)  # Normalize to 0-1
        
        return efficiency
    
    def run_full_analysis(self, items: List[str], model_type: str = 'lightgbm') -> Dict[str, Any]:
        """
        Run complete predictive market analysis including training, prediction, 
        simulation, and insight generation using file data.
        """
        
        print("Starting comprehensive market analysis...")
        
        try:
            # Step 1: Train ML models
            print("Step 1: Training ML models...")
            training_results = self.train_predictive_models_from_files(items, model_type)
            
            # Step 2: Generate predictions
            print("Step 2: Generating ML predictions...")
            predictions = self.generate_ml_predictions_from_files(items)
            
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
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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