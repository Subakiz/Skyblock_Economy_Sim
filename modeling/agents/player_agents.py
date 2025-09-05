"""
Phase 3: SkyBlock Virtual Player Agents
Implementation of different player archetypes for agent-based modeling.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from mesa import Agent
from dataclasses import dataclass
from enum import Enum

class PlayerArchetype(Enum):
    """Different types of SkyBlock players."""
    EARLY_GAME_FARMER = "early_game_farmer"
    MID_GAME_CRAFTER = "mid_game_crafter"
    END_GAME_DUNGEON_RUNNER = "end_game_dungeon_runner"
    AUCTION_FLIPPER = "auction_flipper"
    WHALE_INVESTOR = "whale_investor"
    CASUAL_PLAYER = "casual_player"

@dataclass
class PlayerStats:
    """Player statistics and capabilities."""
    skill_level: int  # 1-60
    net_worth: float  # Total coins + item values
    daily_playtime: float  # Hours per day
    risk_tolerance: float  # 0.0 (conservative) to 1.0 (aggressive)
    market_knowledge: float  # 0.0 (novice) to 1.0 (expert)
    reaction_speed: float  # 0.0 (slow) to 1.0 (instant)

class SkyBlockAgent(Agent):
    """Base class for all SkyBlock player agents."""
    
    def __init__(self, model, unique_id: int, archetype: PlayerArchetype, stats: PlayerStats):
        super().__init__(model)  # Mesa 3.x takes model first
        self.unique_id = unique_id
        self.archetype = archetype
        self.stats = stats
        self.inventory = {}  # item_id -> quantity
        self.coins = stats.net_worth * 0.1  # Start with 10% as liquid coins
        self.transactions = []  # Transaction history
        self.current_activities = []  # Current in-game activities
        
        # Behavioral parameters
        self.decision_cooldown = 0  # Steps until next decision
        self.price_memory = {}  # item_id -> recent price observations
        self.profit_target = self._calculate_profit_target()
        
    def _calculate_profit_target(self) -> float:
        """Calculate profit target based on archetype and stats."""
        base_targets = {
            PlayerArchetype.EARLY_GAME_FARMER: 0.05,  # 5% profit
            PlayerArchetype.MID_GAME_CRAFTER: 0.15,   # 15% profit
            PlayerArchetype.END_GAME_DUNGEON_RUNNER: 0.20,  # 20% profit
            PlayerArchetype.AUCTION_FLIPPER: 0.10,    # 10% profit (high volume)
            PlayerArchetype.WHALE_INVESTOR: 0.25,     # 25% profit (patient)
            PlayerArchetype.CASUAL_PLAYER: 0.08,      # 8% profit
        }
        
        base = base_targets.get(self.archetype, 0.10)
        # Adjust based on risk tolerance and market knowledge
        adjustment = (self.stats.risk_tolerance - 0.5) * 0.1 + (self.stats.market_knowledge - 0.5) * 0.05
        return base + adjustment
    
    def observe_market(self, item_id: str, current_price: float):
        """Update agent's price memory."""
        if item_id not in self.price_memory:
            self.price_memory[item_id] = []
        
        self.price_memory[item_id].append(current_price)
        # Keep only recent observations (based on reaction speed)
        max_memory = int(10 * self.stats.reaction_speed) + 5
        self.price_memory[item_id] = self.price_memory[item_id][-max_memory:]
    
    def get_price_trend(self, item_id: str) -> Optional[float]:
        """Get price trend for an item (-1 to 1, negative = declining)."""
        if item_id not in self.price_memory or len(self.price_memory[item_id]) < 3:
            return None
        
        prices = self.price_memory[item_id]
        # Simple linear trend
        x = np.arange(len(prices))
        y = np.array(prices)
        trend = np.polyfit(x, y, 1)[0]  # Slope of trend line
        
        # Normalize by current price
        return trend / prices[-1] if prices[-1] > 0 else 0
    
    @abstractmethod
    def make_decision(self, market_state: Dict) -> List[Dict]:
        """Make trading/activity decisions based on current market state."""
        pass
    
    def step(self):
        """Agent step in the simulation."""
        if self.decision_cooldown > 0:
            self.decision_cooldown -= 1
            return
        
        # Get market state from model
        market_state = self.model.get_market_state()
        
        # Make decisions
        decisions = self.make_decision(market_state)
        
        # Execute decisions
        for decision in decisions:
            self.execute_decision(decision)
        
        # Set cooldown based on archetype
        self.decision_cooldown = self._get_decision_cooldown()
    
    def execute_decision(self, decision: Dict):
        """Execute a trading decision."""
        action = decision.get('action')
        
        if action == 'buy':
            self._execute_buy(decision)
        elif action == 'sell':
            self._execute_sell(decision)
        elif action == 'craft':
            self._execute_craft(decision)
        elif action == 'hold':
            pass  # No action
    
    def _execute_buy(self, decision: Dict):
        """Execute a buy order."""
        item_id = decision['item_id']
        quantity = decision['quantity']
        max_price = decision['max_price']
        
        market_price = self.model.get_item_price(item_id)
        
        if market_price <= max_price:
            total_cost = market_price * quantity
            if self.coins >= total_cost:
                # Execute purchase
                self.coins -= total_cost
                self.inventory[item_id] = self.inventory.get(item_id, 0) + quantity
                
                # Record transaction
                transaction = {
                    'type': 'buy',
                    'item_id': item_id,
                    'quantity': quantity,
                    'price': market_price,
                    'timestamp': self.model.current_step
                }
                self.transactions.append(transaction)
                
                # Notify market
                self.model.record_transaction(transaction)
    
    def _execute_sell(self, decision: Dict):
        """Execute a sell order."""
        item_id = decision['item_id']
        quantity = decision['quantity']
        min_price = decision['min_price']
        
        market_price = self.model.get_item_price(item_id)
        available = self.inventory.get(item_id, 0)
        
        if market_price >= min_price and available >= quantity:
            # Execute sale
            total_revenue = market_price * quantity
            self.coins += total_revenue
            self.inventory[item_id] -= quantity
            
            # Record transaction
            transaction = {
                'type': 'sell',
                'item_id': item_id,
                'quantity': quantity,
                'price': market_price,
                'timestamp': self.model.current_step
            }
            self.transactions.append(transaction)
            
            # Notify market
            self.model.record_transaction(transaction)
    
    def _execute_craft(self, decision: Dict):
        """Execute a crafting decision."""
        item_id = decision['item_id']
        quantity = decision['quantity']
        recipe = self.model.get_recipe(item_id)
        
        if not recipe:
            return
        
        # Check if we have materials
        can_craft = True
        for material, needed in recipe.items():
            available = self.inventory.get(material, 0)
            if available < needed * quantity:
                can_craft = False
                break
        
        if can_craft:
            # Consume materials
            for material, needed in recipe.items():
                self.inventory[material] -= needed * quantity
            
            # Add crafted items
            self.inventory[item_id] = self.inventory.get(item_id, 0) + quantity
    
    def _get_decision_cooldown(self) -> int:
        """Get decision cooldown based on archetype."""
        cooldowns = {
            PlayerArchetype.EARLY_GAME_FARMER: 10,     # Slow, methodical
            PlayerArchetype.MID_GAME_CRAFTER: 5,       # Moderate speed
            PlayerArchetype.END_GAME_DUNGEON_RUNNER: 8, # Focused on dungeons
            PlayerArchetype.AUCTION_FLIPPER: 1,        # Very fast
            PlayerArchetype.WHALE_INVESTOR: 20,        # Patient, long-term
            PlayerArchetype.CASUAL_PLAYER: 15,         # Infrequent decisions
        }
        
        base_cooldown = cooldowns.get(self.archetype, 5)
        # Adjust based on reaction speed
        return max(1, int(base_cooldown * (2 - self.stats.reaction_speed)))

class EarlyGameFarmer(SkyBlockAgent):
    """Early game player focused on farming and basic crafting."""
    
    def __init__(self, model, unique_id: int):
        stats = PlayerStats(
            skill_level=random.randint(1, 25),
            net_worth=random.uniform(50000, 500000),  # 50k - 500k coins
            daily_playtime=random.uniform(2, 6),
            risk_tolerance=random.uniform(0.1, 0.4),
            market_knowledge=random.uniform(0.1, 0.3),
            reaction_speed=random.uniform(0.2, 0.5)
        )
        super().__init__(model, unique_id, PlayerArchetype.EARLY_GAME_FARMER, stats)
        
        # Focus on basic farming items
        self.focus_items = ['WHEAT', 'CARROT', 'POTATO', 'MELON', 'PUMPKIN', 
                           'ENCHANTED_BREAD', 'ENCHANTED_CARROT', 'ENCHANTED_POTATO']
    
    def make_decision(self, market_state: Dict) -> List[Dict]:
        """Simple farming-focused decisions."""
        decisions = []
        
        # Look for profitable farming items
        for item_id in self.focus_items:
            if item_id in market_state['prices']:
                current_price = market_state['prices'][item_id]
                self.observe_market(item_id, current_price)
                
                trend = self.get_price_trend(item_id)
                
                # Simple buy low, sell high strategy
                if trend is not None and trend > 0.05 and current_price < self.coins * 0.1:
                    # Price trending up, consider buying
                    quantity = min(64, int(self.coins * 0.05 / current_price))
                    if quantity > 0:
                        decisions.append({
                            'action': 'buy',
                            'item_id': item_id,
                            'quantity': quantity,
                            'max_price': current_price * 1.02  # 2% slippage tolerance
                        })
                
                # Sell if we have inventory and price is good
                available = self.inventory.get(item_id, 0)
                if available > 10 and trend is not None and trend < -0.03:
                    # Price declining, sell
                    sell_quantity = min(available, available // 2)  # Sell half
                    decisions.append({
                        'action': 'sell',
                        'item_id': item_id,
                        'quantity': sell_quantity,
                        'min_price': current_price * 0.98  # Accept 2% below market
                    })
        
        return decisions

class EndGameDungeonRunner(SkyBlockAgent):
    """End game player focused on dungeon items and high-value trades."""
    
    def __init__(self, model, unique_id: int):
        stats = PlayerStats(
            skill_level=random.randint(40, 60),
            net_worth=random.uniform(50000000, 1000000000),  # 50M - 1B coins
            daily_playtime=random.uniform(4, 12),
            risk_tolerance=random.uniform(0.4, 0.8),
            market_knowledge=random.uniform(0.6, 0.9),
            reaction_speed=random.uniform(0.6, 0.9)
        )
        super().__init__(model, unique_id, PlayerArchetype.END_GAME_DUNGEON_RUNNER, stats)
        
        # Focus on high-value dungeon items
        self.focus_items = ['NECRON_CHESTPLATE', 'HYPERION', 'WITHER_SKULL', 
                           'SHADOW_ASSASSIN_CHESTPLATE', 'LIVID_DAGGER']
    
    def make_decision(self, market_state: Dict) -> List[Dict]:
        """Strategic decisions for high-value items."""
        decisions = []
        
        for item_id in self.focus_items:
            if item_id in market_state['prices']:
                current_price = market_state['prices'][item_id]
                self.observe_market(item_id, current_price)
                
                trend = self.get_price_trend(item_id)
                
                # More sophisticated trading logic
                if trend is not None:
                    # Consider market volatility and own position
                    position_value = self.inventory.get(item_id, 0) * current_price
                    portfolio_weight = position_value / max(self.coins + position_value, 1)
                    
                    # Buy if trending up and don't have too much exposure
                    if trend > 0.02 and portfolio_weight < 0.3:
                        max_spend = self.coins * 0.2  # Max 20% of coins
                        if current_price < max_spend:
                            quantity = max(1, int(max_spend / current_price * 0.1))
                            decisions.append({
                                'action': 'buy',
                                'item_id': item_id,
                                'quantity': quantity,
                                'max_price': current_price * 1.05
                            })
                    
                    # Sell if trending down or have high exposure
                    elif trend < -0.02 or portfolio_weight > 0.4:
                        available = self.inventory.get(item_id, 0)
                        if available > 0:
                            sell_quantity = max(1, available // 4)  # Sell 25%
                            decisions.append({
                                'action': 'sell',
                                'item_id': item_id,
                                'quantity': sell_quantity,
                                'min_price': current_price * 0.95
                            })
        
        return decisions

class AuctionFlipper(SkyBlockAgent):
    """Player focused on rapid auction house flipping."""
    
    def __init__(self, model, unique_id: int):
        stats = PlayerStats(
            skill_level=random.randint(20, 50),
            net_worth=random.uniform(5000000, 100000000),  # 5M - 100M coins
            daily_playtime=random.uniform(6, 15),
            risk_tolerance=random.uniform(0.3, 0.7),
            market_knowledge=random.uniform(0.7, 1.0),
            reaction_speed=random.uniform(0.8, 1.0)
        )
        super().__init__(model, unique_id, PlayerArchetype.AUCTION_FLIPPER, stats)
        
        # Monitor many items for flip opportunities
        self.focus_items = ['ENCHANTED_LAPIS_BLOCK', 'ENCHANTED_REDSTONE_BLOCK',
                           'ENCHANTED_IRON_BLOCK', 'ENCHANTED_GOLD_BLOCK',
                           'ASPECT_OF_THE_DRAGON', 'STRONG_DRAGON_CHESTPLATE']
    
    def make_decision(self, market_state: Dict) -> List[Dict]:
        """Fast-paced flipping decisions."""
        decisions = []
        
        for item_id in self.focus_items:
            if item_id in market_state['prices']:
                current_price = market_state['prices'][item_id]
                self.observe_market(item_id, current_price)
                
                # Look for short-term price movements
                if len(self.price_memory[item_id]) >= 3:
                    recent_prices = self.price_memory[item_id][-3:]
                    price_volatility = np.std(recent_prices) / np.mean(recent_prices)
                    
                    # High volatility = opportunity
                    if price_volatility > 0.05:
                        current_position = self.inventory.get(item_id, 0)
                        
                        # Quick flip logic
                        if current_price < recent_prices[-2] * 0.95 and current_position == 0:
                            # Price dropped, buy for flip
                            max_quantity = int(self.coins * 0.1 / current_price)
                            if max_quantity > 0:
                                decisions.append({
                                    'action': 'buy',
                                    'item_id': item_id,
                                    'quantity': min(max_quantity, 10),  # Limit risk
                                    'max_price': current_price * 1.01
                                })
                        
                        elif current_price > recent_prices[-2] * 1.05 and current_position > 0:
                            # Price jumped, sell
                            decisions.append({
                                'action': 'sell',
                                'item_id': item_id,
                                'quantity': current_position,
                                'min_price': current_price * 0.99
                            })
        
        return decisions

def create_agent_population(model, total_agents: int = 100) -> List[SkyBlockAgent]:
    """Create a diverse population of agents."""
    agents = []
    
    # Distribution of agent types (realistic SkyBlock population)
    distribution = {
        EarlyGameFarmer: 0.40,        # 40% early game farmers
        EndGameDungeonRunner: 0.15,   # 15% end game players  
        AuctionFlipper: 0.10,         # 10% flippers
    }
    
    agent_id = 0
    for agent_class, proportion in distribution.items():
        count = int(total_agents * proportion)
        for _ in range(count):
            agents.append(agent_class(model, agent_id))
            agent_id += 1
    
    # Fill remaining with casual players
    while len(agents) < total_agents:
        stats = PlayerStats(
            skill_level=random.randint(10, 40),
            net_worth=random.uniform(100000, 10000000),
            daily_playtime=random.uniform(1, 4),
            risk_tolerance=random.uniform(0.2, 0.6),
            market_knowledge=random.uniform(0.2, 0.5),
            reaction_speed=random.uniform(0.3, 0.7)
        )
        agent = SkyBlockAgent(model, agent_id, PlayerArchetype.CASUAL_PLAYER, stats)
        
        # Simple casual behavior
        def make_decision(self, market_state):
            return []  # Mostly inactive
        
        agent.make_decision = make_decision.__get__(agent, SkyBlockAgent)
        agents.append(agent)
        agent_id += 1
    
    return agents