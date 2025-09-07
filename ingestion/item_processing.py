#!/usr/bin/env python3
"""
Item Processing Module for Canonical Name Generation
Handles parsing of auction items to create specific canonical names that include attributes.
"""

import re
import logging
from typing import Dict, Set, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ItemProcessor:
    """Processes auction items to generate canonical names with attributes."""
    
    def __init__(self):
        """Initialize the item processor with attribute patterns."""
        # Pet held item patterns
        self.held_item_patterns = [
            r'Held Item: ([A-Za-z0-9\s_]+)',
            r'Pet Item: ([A-Za-z0-9\s_]+)',
        ]
        
        # Weapon enchantment patterns
        self.ultimate_enchant_patterns = [
            r'Ultimate ([A-Za-z\s]+) [IVX]+',
            r'Ultimate Enchantment: ([A-Za-z\s]+)',
        ]
        
        # Armor/weapon star patterns  
        self.star_patterns = [
            r'⚚⚚⚚⚚⚚',  # 5 stars
            r'⚚⚚⚚⚚',     # 4 stars
            r'⚚⚚⚚',       # 3 stars
            r'⚚⚚',         # 2 stars
            r'⚚',          # 1 star
        ]
        
        # Reforge patterns
        self.beneficial_reforges = {
            'weapons': ['Legendary', 'Fabled', 'Renowned', 'Spicy', 'Sharp', 'Heroic', 'Epic'],
            'armor': ['Pure', 'Perfect', 'Necrotic', 'Ancient', 'Renowned', 'Giant', 'Smart']
        }
        
        # Item type keywords
        self.weapon_keywords = [
            'SWORD', 'BOW', 'HYPERION', 'VALKYRIE', 'SCYLLA', 'NECRON_BLADE', 
            'GIANTS_SWORD', 'LIVID_DAGGER', 'SHADOW_FURY', 'ASPECT_OF_THE',
            'MIDAS_SWORD', 'EMERALD_BLADE', 'LEAPING_SWORD'
        ]
        
        self.armor_keywords = [
            'HELMET', 'CHESTPLATE', 'LEGGINGS', 'BOOTS', 'NECRON', 'STORM',
            'GOLDOR', 'MAXOR', 'SUPERIOR', 'ELEGANT_TUXEDO', 'FROZEN_BLAZE'
        ]
        
        self.pet_keywords = ['[Lvl', 'Pet', 'WOLF', 'DRAGON', 'LION', 'RABBIT', 'TURTLE', 'BEE', 'ENDERMAN', 'PIGMAN']

    def create_canonical_name(self, item_name: str, item_lore: str = "", nbt_data: Optional[Dict] = None) -> str:
        """
        Create a canonical name for an item that includes its important attributes.
        
        Args:
            item_name: The base item name from auction
            item_lore: The item's lore text
            nbt_data: Optional NBT data dictionary
            
        Returns:
            Canonical name with attributes (e.g., "[Lvl 100] Wolf (Held Item: Combat Exp Boost)")
        """
        try:
            # Start with the original item name
            canonical = item_name.strip()
            attributes = []
            
            # Parse pet held items
            if self._is_pet_item(canonical):
                held_item = self._extract_held_item(item_lore)
                if held_item:
                    attributes.append(f"Held Item: {held_item}")
            
            # Parse weapon attributes
            elif self._is_weapon_item(canonical):
                # Check for ultimate enchantments
                ultimate = self._extract_ultimate_enchant(item_lore)
                if ultimate:
                    attributes.append(f"Ultimate: {ultimate}")
                
                # Check for good reforges
                reforge = self._extract_beneficial_reforge(item_lore, 'weapons')
                if reforge:
                    attributes.append(f"Reforge: {reforge}")
                    
                # Check for stars
                stars = self._extract_star_count(item_lore)
                if stars > 0:
                    attributes.append(f"Stars: {stars}")
            
            # Parse armor attributes
            elif self._is_armor_item(canonical):
                # Check for good reforges
                reforge = self._extract_beneficial_reforge(item_lore, 'armor')
                if reforge:
                    attributes.append(f"Reforge: {reforge}")
                    
                # Check for stars
                stars = self._extract_star_count(item_lore)
                if stars > 0:
                    attributes.append(f"Stars: {stars}")
            
            # Add attributes to canonical name
            if attributes:
                canonical = f"{canonical} ({', '.join(attributes)})"
                
            return canonical
            
        except Exception as e:
            logger.error(f"Error creating canonical name for '{item_name}': {e}")
            return item_name.strip()  # Fallback to original name

    def _is_pet_item(self, item_name: str) -> bool:
        """Check if item is a pet."""
        item_upper = item_name.upper()
        return any(keyword in item_upper for keyword in self.pet_keywords)

    def _is_weapon_item(self, item_name: str) -> bool:
        """Check if item is a weapon."""
        item_upper = item_name.upper()
        return any(keyword in item_upper for keyword in self.weapon_keywords)

    def _is_armor_item(self, item_name: str) -> bool:
        """Check if item is armor."""
        item_upper = item_name.upper()
        return any(keyword in item_upper for keyword in self.armor_keywords)

    def _extract_held_item(self, lore: str) -> Optional[str]:
        """Extract held item from pet lore."""
        if not lore:
            return None
            
        for pattern in self.held_item_patterns:
            match = re.search(pattern, lore, re.IGNORECASE)
            if match:
                held_item = match.group(1).strip()
                # Clean up common formatting
                held_item = held_item.replace('§', '').strip()
                if held_item and held_item != 'None':
                    return held_item
        return None

    def _extract_ultimate_enchant(self, lore: str) -> Optional[str]:
        """Extract ultimate enchantment from item lore.""" 
        if not lore:
            return None
            
        for pattern in self.ultimate_enchant_patterns:
            match = re.search(pattern, lore, re.IGNORECASE)
            if match:
                enchant = match.group(1).strip()
                # Clean up formatting
                enchant = enchant.replace('§', '').strip()
                if enchant:
                    return enchant
        return None

    def _extract_beneficial_reforge(self, lore: str, item_type: str) -> Optional[str]:
        """Extract beneficial reforge from item lore."""
        if not lore or item_type not in self.beneficial_reforges:
            return None
            
        beneficial = self.beneficial_reforges[item_type]
        
        for reforge in beneficial:
            # Look for reforge name in lore (case-insensitive)
            if re.search(rf'\b{re.escape(reforge)}\b', lore, re.IGNORECASE):
                return reforge
        return None

    def _extract_star_count(self, lore: str) -> int:
        """Extract star count from item lore."""
        if not lore:
            return 0
            
        # Check each star pattern and return the highest match
        for i, pattern in enumerate(self.star_patterns):
            if re.search(pattern, lore):
                return 5 - i  # Convert index to star count (5, 4, 3, 2, 1)
        return 0

    def is_item_valuable_variant(self, canonical_name: str) -> bool:
        """
        Check if an item has valuable attributes that differentiate it from base item.
        Used to determine if this variant should be tracked separately.
        """
        # Items with attributes in parentheses are valuable variants
        return '(' in canonical_name and ')' in canonical_name

    def get_base_item_name(self, canonical_name: str) -> str:
        """
        Extract the base item name from a canonical name.
        Example: "[Lvl 100] Wolf (Held Item: Combat Exp Boost)" -> "[Lvl 100] Wolf"
        """
        if '(' in canonical_name:
            return canonical_name.split('(')[0].strip()
        return canonical_name

    def get_attribute_signature(self, canonical_name: str) -> str:
        """
        Extract the attribute signature from a canonical name.
        Example: "[Lvl 100] Wolf (Held Item: Combat Exp Boost)" -> "Held Item: Combat Exp Boost"
        """
        if '(' in canonical_name and ')' in canonical_name:
            return canonical_name.split('(', 1)[1].rsplit(')', 1)[0]
        return ""


# Global instance for easy importing
item_processor = ItemProcessor()


def create_canonical_name(item_name: str, item_lore: str = "", nbt_data: Optional[Dict] = None) -> str:
    """
    Convenience function to create canonical names.
    This is the main function that should be used by the ingestion pipeline.
    """
    return item_processor.create_canonical_name(item_name, item_lore, nbt_data)


def get_base_item_name(canonical_name: str) -> str:
    """Convenience function to get base item name."""
    return item_processor.get_base_item_name(canonical_name)


def is_valuable_variant(canonical_name: str) -> bool:
    """Convenience function to check if item is a valuable variant."""
    return item_processor.is_item_valuable_variant(canonical_name)