#!/usr/bin/env python3
"""
Phase 3 Demonstration Script
Shows the capabilities of the advanced ML and agent-based modeling system.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from modeling.agents.market_simulation import ScenarioEngine, MarketModel
    from modeling.agents.player_agents import PlayerArchetype
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"Phase 3 features not available: {e}")
    PHASE3_AVAILABLE = False

def demo_agent_archetypes():
    """Demonstrate different agent archetypes."""
    print("\n🤖 Agent Archetypes Demonstration")
    print("=" * 50)
    
    archetypes = [
        ("Early Game Farmer", "Low-risk farming and basic items (40% of population)"),
        ("End Game Dungeon Runner", "High-value dungeon items and sophisticated trading (15%)"),
        ("Auction Flipper", "Rapid trading and market-making behavior (10%)"),
        ("Casual Player", "Infrequent activity with simple strategies (35%)")
    ]
    
    for archetype, description in archetypes:
        print(f"• {archetype}: {description}")

def demo_market_scenarios():
    """Demonstrate available market scenarios."""
    print("\n📊 Market Scenarios Demonstration")
    print("=" * 50)
    
    if not PHASE3_AVAILABLE:
        print("Phase 3 not available for scenario demonstration.")
        return
    
    engine = ScenarioEngine()
    
    print("Available market scenarios:")
    for name, config in engine.scenarios.items():
        volatility = config.get('volatility', 'N/A')
        agents = config.get('n_agents', 'N/A')
        print(f"• {name}: {config['description']}")
        print(f"  Volatility: {volatility}, Default agents: {agents}")

def demo_quick_simulation():
    """Run a quick demonstration simulation."""
    print("\n🎮 Quick Market Simulation")
    print("=" * 50)
    
    if not PHASE3_AVAILABLE:
        print("Phase 3 not available for simulation demonstration.")
        return
    
    print("Running a quick 20-step simulation with 20 agents...")
    
    try:
        model = MarketModel(n_agents=20, market_volatility=0.05, external_shocks=True)
        results = model.run_simulation(steps=20, verbose=False)
        
        print(f"✓ Simulation completed!")
        print(f"  Final market sentiment: {results['final_sentiment']:.3f}")
        print(f"  Total transactions: {results['transaction_count']}")
        print(f"  Active agents: {len([a for a in results['agent_performance'] if a['transactions'] > 0])}")
        
        # Show top performing agents
        top_performers = sorted(results['agent_performance'], 
                              key=lambda x: x['performance_pct'], reverse=True)[:3]
        
        print(f"\nTop performing agents:")
        for i, agent in enumerate(top_performers, 1):
            print(f"  {i}. {agent['archetype']}: {agent['performance_pct']:.1f}% return "
                  f"({agent['transactions']} trades)")
        
        # Show significant price changes
        significant_changes = {item: change for item, change in results['price_changes'].items() 
                             if abs(change) > 5}
        
        if significant_changes:
            print(f"\nSignificant price movements:")
            for item, change in list(significant_changes.items())[:3]:
                direction = "↑" if change > 0 else "↓"
                print(f"  {direction} {item}: {change:+.1f}%")
    
    except Exception as e:
        print(f"Simulation failed: {e}")

def demo_ml_features():
    """Demonstrate ML forecasting features."""
    print("\n🧠 Machine Learning Features")
    print("=" * 50)
    
    features = [
        "Multivariate LightGBM and XGBoost models",
        "16+ advanced features (momentum, MA ratios, market context)",
        "Multi-horizon predictions (15min, 60min, 240min)",
        "Cross-item dependency modeling",
        "Time series cross-validation",
        "Feature importance analysis"
    ]
    
    print("Advanced ML capabilities:")
    for feature in features:
        print(f"• {feature}")
    
    print(f"\nExample features used:")
    example_features = [
        "ma_15, ma_60 (moving averages)",
        "momentum_1, momentum_5 (price momentum)",
        "vol_price_ratio (volatility-price relationship)",  
        "market_volatility (market-wide context)",
        "hour_of_day, day_of_week (temporal patterns)"
    ]
    
    for feature in example_features:
        print(f"  - {feature}")

def demo_integration_workflow():
    """Demonstrate the integrated ML + ABM workflow."""
    print("\n🔗 Integrated Prediction Workflow")
    print("=" * 50)
    
    workflow_steps = [
        ("1. Data Collection", "Historical price data with 500+ points per item"),
        ("2. Feature Engineering", "16+ multivariate features including cross-item effects"),
        ("3. ML Model Training", "LightGBM/XGBoost with time series validation"),
        ("4. Price Predictions", "Generate forecasts for 15min, 60min, 240min horizons"),
        ("5. Agent Simulation", "Use predictions as initial conditions for ABM"),
        ("6. Scenario Testing", "Test multiple market conditions and shocks"),
        ("7. Insight Generation", "Identify trading opportunities and risk assessment"),
        ("8. Proactive Alerts", "Generate actionable trading signals")
    ]
    
    for step, description in workflow_steps:
        print(f"{step}: {description}")

def main():
    """Run the complete demonstration."""
    print("🌟 SkyBlock Economic Modeling - Phase 3 Demonstration")
    print("=" * 80)
    print("Transforming reactive analysis into proactive market prediction!")
    
    demo_agent_archetypes()
    demo_market_scenarios()
    demo_ml_features()
    demo_integration_workflow()
    demo_quick_simulation()
    
    print("\n" + "=" * 80)
    print("🎯 Phase 3 Capabilities Summary:")
    print("• Multivariate ML models (LightGBM/XGBoost) for price prediction")
    print("• Agent-based modeling with realistic player archetypes")
    print("• Market simulation with external shocks and sentiment tracking")
    print("• Integrated prediction engine combining ML + ABM")
    print("• Comprehensive API and CLI interfaces")
    print("• Scenario testing and market insight generation")
    
    if PHASE3_AVAILABLE:
        print("\n✅ Phase 3 is fully operational!")
        print("\nTry these commands:")
        print("  ./phase3_cli.py status                    # Check system status")
        print("  ./phase3_cli.py scenarios                 # List scenarios")
        print("  ./phase3_cli.py simulate --agents 100     # Run simulation")
        print("  ./phase3_cli.py compare normal_market,volatile_market")
    else:
        print("\n⚠️  Phase 3 requires additional dependencies:")
        print("   pip install lightgbm xgboost scikit-learn mesa")
    
    print("\nPhase 3: From Reactive Analysis → Proactive Market Intelligence ✨")

if __name__ == "__main__":
    main()