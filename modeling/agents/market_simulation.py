"""
Phase 3: SkyBlock Market Simulation
Agent-based model of the SkyBlock economy with realistic market dynamics.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from mesa import Model, Agent
from mesa import DataCollector
from datetime import datetime, timedelta
import json

from .player_agents import create_agent_population, SkyBlockAgent

class MarketModel(Model):
    """Main market simulation model."""
    
    def __init__(self, 
                 n_agents: int = 100,
                 initial_prices: Optional[Dict[str, float]] = None,
                 market_volatility: float = 0.02,
                 external_shocks: bool = True):
        
        # Initialize Mesa Model (no arguments in Mesa 3.x)
        super().__init__()
        
        self.n_agents = n_agents
        self.market_volatility = market_volatility
        self.external_shocks = external_shocks
        self.current_step = 0
        
        # Market state
        self.item_prices = initial_prices or self._get_default_prices()
        self.price_history = {item: [price] for item, price in self.item_prices.items()}
        self.transaction_log = []
        self.market_sentiment = 0.0  # -1 (bearish) to 1 (bullish)
        
        # Market mechanics
        self.supply_demand = {item: {'supply': 1000, 'demand': 1000} 
                             for item in self.item_prices.keys()}
        self.volume_history = {item: [] for item in self.item_prices.keys()}
        
        # External factors
        self.game_events = self._initialize_game_events()
        self.seasonal_effects = {}
        
        # Crafting recipes (simplified)
        self.recipes = self._load_recipes()
        
        # Create agents using Mesa 3.x approach
        self.agent_list = []  # Use different name to avoid Mesa 3.x conflict
        agents = create_agent_population(self, n_agents)
        
        for agent in agents:
            self.agent_list.append(agent)
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Step": "current_step",
                "Market_Sentiment": "market_sentiment",
                "Total_Volume": self._get_total_volume,
                "Price_Volatility": self._get_price_volatility,
                "Active_Agents": self._count_active_agents
            },
            agent_reporters={
                "Coins": "coins",
                "Inventory_Value": self._agent_inventory_value,
                "Archetype": lambda a: a.archetype.value
            }
        )
    
    def _get_default_prices(self) -> Dict[str, float]:
        """Default item prices for simulation."""
        return {
            # Basic items
            'WHEAT': 2.5,
            'CARROT': 2.0,
            'POTATO': 2.2,
            'MELON': 3.0,
            'PUMPKIN': 4.5,
            
            # Enchanted items
            'ENCHANTED_BREAD': 320,
            'ENCHANTED_CARROT': 640,
            'ENCHANTED_POTATO': 512,
            'ENCHANTED_LAPIS_BLOCK': 25600,
            'ENCHANTED_REDSTONE_BLOCK': 12800,
            'ENCHANTED_IRON_BLOCK': 51200,
            'ENCHANTED_GOLD_BLOCK': 204800,
            
            # High-value items
            'NECRON_CHESTPLATE': 150000000,
            'HYPERION': 750000000,
            'WITHER_SKULL': 25000000,
            'SHADOW_ASSASSIN_CHESTPLATE': 8000000,
            'LIVID_DAGGER': 12000000,
            'ASPECT_OF_THE_DRAGON': 5000000,
            'STRONG_DRAGON_CHESTPLATE': 3500000,
        }
    
    def _initialize_game_events(self) -> List[Dict]:
        """Initialize potential game events that affect the market."""
        return [
            {'type': 'dungeon_event', 'probability': 0.01, 'effect': 0.15, 
             'affected_items': ['NECRON_CHESTPLATE', 'HYPERION', 'WITHER_SKULL']},
            {'type': 'farming_event', 'probability': 0.02, 'effect': 0.10,
             'affected_items': ['WHEAT', 'CARROT', 'POTATO']},
            {'type': 'update_announcement', 'probability': 0.005, 'effect': 0.25,
             'affected_items': ['HYPERION', 'NECRON_CHESTPLATE']},
            {'type': 'mayor_election', 'probability': 0.001, 'effect': 0.20,
             'affected_items': list(self._get_default_prices().keys())},
        ]
    
    def _load_recipes(self) -> Dict[str, Dict[str, int]]:
        """Load crafting recipes (simplified)."""
        return {
            'ENCHANTED_BREAD': {'WHEAT': 128},
            'ENCHANTED_CARROT': {'CARROT': 160},
            'ENCHANTED_POTATO': {'POTATO': 160},
            'ENCHANTED_LAPIS_BLOCK': {'LAPIS_LAZULI': 1280},
            'ENCHANTED_REDSTONE_BLOCK': {'REDSTONE': 1280},
            'ENCHANTED_IRON_BLOCK': {'IRON_INGOT': 1280},
            'ENCHANTED_GOLD_BLOCK': {'GOLD_INGOT': 1280},
        }
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market state for agents."""
        return {
            'prices': self.item_prices.copy(),
            'sentiment': self.market_sentiment,
            'volume': {item: sum(vol[-10:]) if vol else 0 
                      for item, vol in self.volume_history.items()},
            'supply_demand': self.supply_demand.copy(),
            'step': self.current_step
        }
    
    def get_item_price(self, item_id: str) -> float:
        """Get current price for an item."""
        return self.item_prices.get(item_id, 0.0)
    
    def get_recipe(self, item_id: str) -> Optional[Dict[str, int]]:
        """Get crafting recipe for an item."""
        return self.recipes.get(item_id)
    
    def record_transaction(self, transaction: Dict):
        """Record a transaction and update market dynamics."""
        self.transaction_log.append(transaction)
        
        item_id = transaction['item_id']
        quantity = transaction['quantity']
        
        # Update supply/demand
        if transaction['type'] == 'buy':
            self.supply_demand[item_id]['demand'] += quantity
            self.supply_demand[item_id]['supply'] = max(0, 
                self.supply_demand[item_id]['supply'] - quantity)
        else:  # sell
            self.supply_demand[item_id]['supply'] += quantity
            self.supply_demand[item_id]['demand'] = max(0,
                self.supply_demand[item_id]['demand'] - quantity)
        
        # Update volume
        if item_id not in self.volume_history:
            self.volume_history[item_id] = []
        self.volume_history[item_id].append(quantity)
    
    def update_prices(self):
        """Update item prices based on market dynamics."""
        
        for item_id in self.item_prices.keys():
            current_price = self.item_prices[item_id]
            
            # Supply/demand pressure
            supply = max(1, self.supply_demand[item_id]['supply'])
            demand = max(1, self.supply_demand[item_id]['demand'])
            pressure = (demand - supply) / (demand + supply)
            
            # Base price change from supply/demand
            price_change = pressure * 0.05  # Max 5% change per step
            
            # Add market volatility
            volatility_change = np.random.normal(0, self.market_volatility)
            
            # Market sentiment effect
            sentiment_effect = self.market_sentiment * 0.02
            
            # Combine all effects
            total_change = price_change + volatility_change + sentiment_effect
            
            # Apply change with bounds
            new_price = current_price * (1 + total_change)
            
            # Prevent negative prices and extreme changes
            new_price = max(new_price, current_price * 0.8)  # Max 20% drop
            new_price = min(new_price, current_price * 1.2)  # Max 20% rise
            
            self.item_prices[item_id] = max(0.01, new_price)
            
            # Update price history
            self.price_history[item_id].append(new_price)
            
            # Keep limited history
            if len(self.price_history[item_id]) > 1000:
                self.price_history[item_id] = self.price_history[item_id][-1000:]
        
        # Decay supply/demand towards equilibrium
        for item_id in self.supply_demand.keys():
            sd = self.supply_demand[item_id]
            sd['supply'] = sd['supply'] * 0.95 + 1000 * 0.05
            sd['demand'] = sd['demand'] * 0.95 + 1000 * 0.05
    
    def apply_external_shocks(self):
        """Apply external market shocks (game events, updates, etc.)."""
        if not self.external_shocks:
            return
        
        for event in self.game_events:
            if random.random() < event['probability']:
                # Event occurs!
                effect = event['effect'] * (random.random() - 0.5) * 2  # -effect to +effect
                
                print(f"Step {self.current_step}: {event['type']} occurs! Effect: {effect:.3f}")
                
                # Apply to affected items
                for item_id in event['affected_items']:
                    if item_id in self.item_prices:
                        self.item_prices[item_id] *= (1 + effect)
                        self.item_prices[item_id] = max(0.01, self.item_prices[item_id])
                
                # Update market sentiment
                self.market_sentiment = np.clip(self.market_sentiment + effect, -1, 1)
    
    def update_sentiment(self):
        """Update overall market sentiment based on recent activity."""
        
        # Calculate recent volume trends
        recent_volumes = []
        for item_id, volumes in self.volume_history.items():
            if volumes:
                recent = sum(volumes[-10:]) if len(volumes) >= 10 else sum(volumes)
                older = sum(volumes[-20:-10]) if len(volumes) >= 20 else sum(volumes[:-10]) if len(volumes) > 10 else recent
                
                if older > 0:
                    volume_trend = (recent - older) / older
                    recent_volumes.append(volume_trend)
        
        # Calculate price trends
        recent_price_trends = []
        for item_id, prices in self.price_history.items():
            if len(prices) >= 5:
                recent_trend = (prices[-1] - prices[-5]) / prices[-5]
                recent_price_trends.append(recent_trend)
        
        # Update sentiment
        volume_sentiment = np.mean(recent_volumes) if recent_volumes else 0
        price_sentiment = np.mean(recent_price_trends) if recent_price_trends else 0
        
        new_sentiment = 0.3 * volume_sentiment + 0.7 * price_sentiment
        
        # Apply momentum and bounds
        self.market_sentiment = np.clip(
            self.market_sentiment * 0.9 + new_sentiment * 0.1,
            -1, 1
        )
    
    def step(self):
        """Execute one step of the simulation."""
        self.current_step += 1
        
        # Apply external shocks first
        self.apply_external_shocks()
        
        # Agents make decisions and act (Mesa 3.x style)
        for agent in self.agent_list:
            agent.step()
        
        # Update market dynamics
        self.update_prices()
        self.update_sentiment()
        
        # Collect data
        self.datacollector.collect(self)
    
    def run_simulation(self, steps: int = 1000, verbose: bool = False):
        """Run the simulation for a specified number of steps."""
        for i in range(steps):
            self.step()
            
            if verbose and i % 100 == 0:
                print(f"Step {i}: Sentiment={self.market_sentiment:.3f}, "
                      f"Active agents={self._count_active_agents()}")
        
        return self.get_simulation_results()
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results."""
        
        # Get collected data
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()
        
        # Calculate final statistics
        final_prices = self.item_prices.copy()
        initial_prices = {item: prices[0] for item, prices in self.price_history.items()}
        
        price_changes = {
            item: (final_prices[item] - initial_prices[item]) / initial_prices[item] * 100
            for item in final_prices.keys()
        }
        
        # Agent performance
        agent_performance = []
        for agent in self.agent_list:
            initial_wealth = agent.stats.net_worth
            current_wealth = agent.coins + sum(
                self.item_prices[item] * quantity 
                for item, quantity in agent.inventory.items()
            )
            performance = (current_wealth - initial_wealth) / initial_wealth * 100
            
            agent_performance.append({
                'id': agent.unique_id,
                'archetype': agent.archetype.value,
                'initial_wealth': initial_wealth,
                'final_wealth': current_wealth,
                'performance_pct': performance,
                'transactions': len(agent.transactions)
            })
        
        return {
            'model_data': model_data,
            'agent_data': agent_data,
            'price_changes': price_changes,
            'agent_performance': agent_performance,
            'transaction_count': len(self.transaction_log),
            'final_sentiment': self.market_sentiment,
            'steps_completed': self.current_step
        }
    
    def save_results(self, filepath: str):
        """Save simulation results to file."""
        results = self.get_simulation_results()
        
        # Convert DataFrames to JSON-serializable format
        results['model_data'] = results['model_data'].to_dict('records')
        results['agent_data'] = results['agent_data'].reset_index().to_dict('records')
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Helper methods for data collection
    def _get_total_volume(self) -> int:
        """Get total trading volume in current step."""
        current_step_transactions = [t for t in self.transaction_log 
                                   if t['timestamp'] == self.current_step]
        return sum(t['quantity'] for t in current_step_transactions)
    
    def _get_price_volatility(self) -> float:
        """Get current market-wide price volatility."""
        volatilities = []
        for item_id, prices in self.price_history.items():
            if len(prices) >= 10:
                recent_prices = prices[-10:]
                volatility = np.std(recent_prices) / np.mean(recent_prices)
                volatilities.append(volatility)
        
        return np.mean(volatilities) if volatilities else 0.0
    
    def _count_active_agents(self) -> int:
        """Count agents that made transactions recently."""
        recent_traders = set()
        recent_transactions = [t for t in self.transaction_log 
                             if self.current_step - t['timestamp'] <= 10]
        
        for transaction in recent_transactions:
            # Would need agent ID in transaction to count properly
            # For now, estimate based on transaction count
            pass
        
        return len([a for a in self.agent_list if len(a.transactions) > 0])
    
    def _agent_inventory_value(self, agent: SkyBlockAgent) -> float:
        """Calculate total inventory value for an agent."""
        total_value = 0
        for item_id, quantity in agent.inventory.items():
            total_value += self.item_prices.get(item_id, 0) * quantity
        return total_value

class ScenarioEngine:
    """Engine for testing various market scenarios."""
    
    def __init__(self):
        self.scenarios = {
            'normal_market': {
                'description': 'Normal market conditions',
                'volatility': 0.02,
                'shocks': True,
                'n_agents': 100
            },
            'volatile_market': {
                'description': 'High volatility market',
                'volatility': 0.08,
                'shocks': True,
                'n_agents': 100
            },
            'stable_market': {
                'description': 'Low volatility, stable market',
                'volatility': 0.005,
                'shocks': False,
                'n_agents': 100
            },
            'major_update': {
                'description': 'Major game update impact',
                'volatility': 0.04,
                'shocks': True,
                'n_agents': 150,
                'custom_events': [
                    {'type': 'major_update', 'probability': 0.02, 'effect': 0.5,
                     'affected_items': ['HYPERION', 'NECRON_CHESTPLATE', 'WITHER_SKULL']}
                ]
            }
        }
    
    def run_scenario(self, scenario_name: str, steps: int = 1000, 
                    custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a specific market scenario."""
        
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name].copy()
        if custom_params:
            scenario.update(custom_params)
        
        print(f"Running scenario: {scenario['description']}")
        
        # Create model with scenario parameters
        model = MarketModel(
            n_agents=scenario['n_agents'],
            market_volatility=scenario['volatility'],
            external_shocks=scenario['shocks']
        )
        
        # Add custom events if specified
        if 'custom_events' in scenario:
            model.game_events.extend(scenario['custom_events'])
        
        # Run simulation
        results = model.run_simulation(steps, verbose=True)
        results['scenario'] = scenario_name
        results['scenario_description'] = scenario['description']
        
        return results
    
    def compare_scenarios(self, scenario_names: List[str], steps: int = 1000) -> Dict[str, Any]:
        """Compare multiple scenarios."""
        
        scenario_results = {}
        
        for scenario_name in scenario_names:
            print(f"\n{'='*50}")
            print(f"Running scenario: {scenario_name}")
            print('='*50)
            
            results = self.run_scenario(scenario_name, steps)
            scenario_results[scenario_name] = results
        
        # Generate comparison summary
        comparison = {
            'scenarios': scenario_results,
            'summary': self._generate_comparison_summary(scenario_results)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, scenario_results: Dict) -> Dict[str, Any]:
        """Generate summary comparing scenario results."""
        
        summary = {
            'price_volatility': {},
            'agent_performance': {},
            'market_sentiment': {},
            'transaction_volume': {}
        }
        
        for scenario_name, results in scenario_results.items():
            # Price volatility
            price_changes = list(results['price_changes'].values())
            summary['price_volatility'][scenario_name] = {
                'mean': np.mean(price_changes),
                'std': np.std(price_changes),
                'max': max(price_changes),
                'min': min(price_changes)
            }
            
            # Agent performance by archetype
            agent_perf = results['agent_performance']
            archetype_performance = {}
            for agent in agent_perf:
                archetype = agent['archetype']
                if archetype not in archetype_performance:
                    archetype_performance[archetype] = []
                archetype_performance[archetype].append(agent['performance_pct'])
            
            summary['agent_performance'][scenario_name] = {
                arch: np.mean(perfs) for arch, perfs in archetype_performance.items()
            }
            
            # Market sentiment
            summary['market_sentiment'][scenario_name] = results['final_sentiment']
            
            # Transaction volume
            summary['transaction_volume'][scenario_name] = results['transaction_count']
        
        return summary