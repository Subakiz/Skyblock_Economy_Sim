#!/usr/bin/env python3
"""
Phase 3: SkyBlock Economic Modeling CLI
Command-line interface for advanced ML and agent-based modeling features.
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from modeling.forecast.ml_forecaster import train_and_forecast_ml
    from modeling.agents.market_simulation import ScenarioEngine
    from modeling.simulation.predictive_engine import PredictiveMarketEngine
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"Phase 3 features not available: {e}")
    PHASE3_AVAILABLE = False

def train_ml_model(args):
    """Train ML model for price forecasting."""
    if not PHASE3_AVAILABLE:
        print("Phase 3 ML features are not available.")
        return
    
    print(f"Training {args.model_type} model for {args.item_id}...")
    
    try:
        train_and_forecast_ml(
            product_id=args.item_id,
            model_type=args.model_type,
            horizons=tuple(args.horizons)
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")

def run_market_simulation(args):
    """Run market simulation."""
    if not PHASE3_AVAILABLE:
        print("Phase 3 simulation features are not available.")
        return
        
    print(f"Running {args.scenario} simulation with {args.agents} agents for {args.steps} steps...")
    
    try:
        engine = ScenarioEngine()
        results = engine.run_scenario(args.scenario, steps=args.steps, 
                                    custom_params={'n_agents': args.agents})
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            # Print summary
            print(f"\nSimulation Summary:")
            print(f"Final market sentiment: {results['final_sentiment']:.3f}")
            print(f"Total transactions: {results['transaction_count']}")
            
            price_changes = results['price_changes']
            if price_changes:
                avg_change = sum(price_changes.values()) / len(price_changes)
                print(f"Average price change: {avg_change:.2f}%")
        
    except Exception as e:
        print(f"Simulation failed: {e}")

def run_predictive_analysis(args):
    """Run predictive market analysis."""
    if not PHASE3_AVAILABLE:
        print("Phase 3 predictive features are not available.")
        return
    
    items = args.items.split(',')
    print(f"Running predictive analysis for items: {items}")
    
    try:
        engine = PredictiveMarketEngine()
        results = engine.run_full_analysis(items, model_type=args.model_type)
        
        if args.output:
            engine.save_analysis(results, args.output)
        else:
            # Print summary
            print(f"\nPredictive Analysis Summary:")
            print(f"Items analyzed: {len(items)}")
            
            insights = results['market_insights']
            outlook = insights['market_outlook']
            print(f"Market outlook: {outlook.get('short_term_trend', 'unknown')}")
            
            opportunities = insights['trading_opportunities']
            print(f"Trading opportunities found: {len(opportunities)}")
            
            for i, opp in enumerate(opportunities[:3], 1):
                print(f"  {i}. {opp['item_id']}: {opp['type']} "
                      f"({opp['expected_return']:.1f}% return, "
                      f"{opp['confidence']:.2f} confidence)")
    
    except Exception as e:
        print(f"Analysis failed: {e}")

def compare_scenarios(args):
    """Compare multiple market scenarios."""
    if not PHASE3_AVAILABLE:
        print("Phase 3 scenario features are not available.")
        return
    
    scenarios = args.scenarios.split(',')
    print(f"Comparing scenarios: {scenarios}")
    
    try:
        engine = ScenarioEngine()
        comparison = engine.compare_scenarios(scenarios, steps=args.steps)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"Comparison saved to {args.output}")
        else:
            # Print summary
            summary = comparison['summary']
            print(f"\nScenario Comparison Summary:")
            
            print(f"\nPrice Volatility:")
            for scenario, metrics in summary['price_volatility'].items():
                print(f"  {scenario}: {metrics['std']:.2f}% std dev")
            
            print(f"\nTransaction Volume:")
            for scenario, volume in summary['transaction_volume'].items():
                print(f"  {scenario}: {volume} transactions")
    
    except Exception as e:
        print(f"Comparison failed: {e}")

def list_scenarios(args):
    """List available scenarios."""
    if not PHASE3_AVAILABLE:
        print("Phase 3 scenario features are not available.")
        return
    
    try:
        engine = ScenarioEngine()
        scenarios = engine.scenarios
        
        print("Available scenarios:")
        for name, config in scenarios.items():
            print(f"  {name}: {config['description']}")
    
    except Exception as e:
        print(f"Failed to list scenarios: {e}")

def check_status(args):
    """Check Phase 3 feature status."""
    print("SkyBlock Economic Modeling - Phase 3 Status")
    print("=" * 50)
    
    print(f"Phase 3 available: {PHASE3_AVAILABLE}")
    
    if PHASE3_AVAILABLE:
        # Check model directory
        model_dir = Path("models")
        if model_dir.exists():
            models = list(model_dir.glob("*.pkl"))
            print(f"Trained models: {len(models)}")
            
            if models:
                print("Recent models:")
                for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    mtime = datetime.fromtimestamp(model.stat().st_mtime)
                    print(f"  {model.name} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("No models directory found")
        
        # Check dependencies
        try:
            import lightgbm
            print(f"LightGBM version: {lightgbm.__version__}")
        except ImportError:
            print("LightGBM: not available")
        
        try:
            import xgboost
            print(f"XGBoost version: {xgboost.__version__}")
        except ImportError:
            print("XGBoost: not available")
        
        try:
            import mesa
            print(f"Mesa version: {mesa.__version__}")
        except ImportError:
            print("Mesa: not available")
    
    else:
        print("Phase 3 features are not available. Please install required dependencies:")
        print("  pip install lightgbm xgboost scikit-learn mesa")

def main():
    parser = argparse.ArgumentParser(description="SkyBlock Economic Modeling - Phase 3 CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ML training command
    train_parser = subparsers.add_parser('train', help='Train ML forecasting model')
    train_parser.add_argument('item_id', help='Item ID to train model for')
    train_parser.add_argument('--model-type', default='lightgbm', 
                             choices=['lightgbm', 'xgboost'],
                             help='ML model type')
    train_parser.add_argument('--horizons', nargs='+', type=int, 
                             default=[15, 60, 240],
                             help='Prediction horizons in minutes')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run market simulation')
    sim_parser.add_argument('--scenario', default='normal_market',
                           help='Simulation scenario')
    sim_parser.add_argument('--agents', type=int, default=100,
                           help='Number of agents')
    sim_parser.add_argument('--steps', type=int, default=500,
                           help='Simulation steps')
    sim_parser.add_argument('--output', help='Output file for results')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Run predictive analysis')
    analysis_parser.add_argument('items', help='Comma-separated list of item IDs')
    analysis_parser.add_argument('--model-type', default='lightgbm',
                                choices=['lightgbm', 'xgboost'],
                                help='ML model type')
    analysis_parser.add_argument('--output', help='Output file for results')
    
    # Scenario comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare scenarios')
    compare_parser.add_argument('scenarios', help='Comma-separated list of scenarios')
    compare_parser.add_argument('--steps', type=int, default=500,
                               help='Simulation steps')
    compare_parser.add_argument('--output', help='Output file for results')
    
    # List scenarios command
    subparsers.add_parser('scenarios', help='List available scenarios')
    
    # Status command
    subparsers.add_parser('status', help='Check Phase 3 status')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_ml_model(args)
    elif args.command == 'simulate':
        run_market_simulation(args)
    elif args.command == 'analyze':
        run_predictive_analysis(args)
    elif args.command == 'compare':
        compare_scenarios(args)
    elif args.command == 'scenarios':
        list_scenarios(args)
    elif args.command == 'status':
        check_status(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()