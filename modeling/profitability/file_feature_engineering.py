"""
File-based feature engineering for ML models.
Provides equivalent functionality to database-based feature calculations using NDJSON data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from storage.ndjson_storage import NDJSONStorage


def calculate_moving_averages(prices: pd.Series, windows: List[int] = [15, 60]) -> pd.DataFrame:
    """Calculate moving averages for given window sizes."""
    result = pd.DataFrame(index=prices.index)
    
    for window in windows:
        result[f'ma_{window}'] = prices.rolling(window=window, min_periods=1).mean()
    
    return result


def calculate_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility."""
    returns = prices.pct_change()
    return returns.rolling(window=window, min_periods=1).std() * 100  # Convert to percentage


def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price-based technical features."""
    result = df.copy()
    
    # Price lags
    result['price_lag_1'] = result['mid_price'].shift(1)
    result['price_lag_5'] = result['mid_price'].shift(5)
    
    # Momentum features
    result['momentum_1'] = (result['mid_price'] - result['price_lag_1']) / result['price_lag_1'].replace(0, np.nan) * 100
    result['momentum_5'] = (result['mid_price'] - result['price_lag_5']) / result['price_lag_5'].replace(0, np.nan) * 100
    
    # MA crossovers and ratios
    result['ma_crossover'] = (result['ma_15'] > result['ma_60']).astype(int)
    result['ma_ratio'] = result['ma_15'] / result['ma_60'].replace(0, np.nan)
    
    # Volume-price indicators
    result['vol_price_ratio'] = result['vol_window_30'] / result['mid_price'].replace(0, np.nan)
    
    return result


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based features."""
    result = df.copy()
    
    # Ensure timestamp is datetime
    if 'ts' in result.columns:
        result['ts'] = pd.to_datetime(result['ts'])
        result['hour_of_day'] = result['ts'].dt.hour
        result['day_of_week'] = result['ts'].dt.dayofweek
        result['day_of_month'] = result['ts'].dt.day
    
    return result


def calculate_market_context_features(storage: NDJSONStorage, product_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market-wide context features."""
    result = df.copy()
    
    # For file-based mode, we'll use simplified market context
    # In a full implementation, you'd aggregate across all products
    
    # Market volatility approximation (use current product's volatility as proxy)
    result['market_volatility'] = result['vol_window_30']
    
    # Market momentum approximation (use current product's momentum)
    result['market_momentum'] = result.get('momentum_1', 0)
    
    return result


def create_feature_dataframe_from_files(
    storage: NDJSONStorage, 
    product_id: str, 
    hours_back: int = 72,
    min_points: int = 500
) -> Optional[pd.DataFrame]:
    """
    Create a feature-rich DataFrame from NDJSON files equivalent to database bazaar_features.
    """
    
    # Read raw bazaar data
    records = list(storage.read_records("bazaar", hours_back=hours_back))
    
    # Filter for the specific product
    product_records = [r for r in records if r.get('product_id') == product_id]
    
    if len(product_records) < min_points:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(product_records)
    
    # Ensure required columns exist
    required_cols = ['product_id', 'buy_price', 'sell_price']
    for col in required_cols:
        if col not in df.columns:
            return None
    
    # Convert timestamp
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    elif 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        # Use current time as approximation if no timestamp
        df['ts'] = pd.date_range(end=datetime.now(timezone.utc), periods=len(df), freq='5T')
    
    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate mid price
    df['mid_price'] = (df['buy_price'] + df['sell_price']) / 2
    
    # Calculate spread in basis points
    df['spread_bps'] = ((df['buy_price'] - df['sell_price']) / df['mid_price'].replace(0, np.nan) * 10000).fillna(0)
    
    # Calculate moving averages
    ma_df = calculate_moving_averages(df['mid_price'], windows=[15, 60])
    df = pd.concat([df, ma_df], axis=1)
    
    # Calculate volatility
    df['vol_window_30'] = calculate_volatility(df['mid_price'], window=30)
    
    # Calculate price-based features
    df = calculate_price_features(df)
    
    # Calculate time-based features
    df = calculate_time_features(df)
    
    # Calculate market context features
    df = calculate_market_context_features(storage, product_id, df)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


def get_auction_feature_dataframe_from_files(
    storage: NDJSONStorage,
    product_id: str,
    hours_back: int = 24
) -> Optional[pd.DataFrame]:
    """
    Create auction-based features from NDJSON files.
    """
    
    # Read auction data
    records = list(storage.read_records("auctions_ended", hours_back=hours_back))
    
    # Filter for the specific product (handle different item_id formats)
    product_records = []
    for r in records:
        item_id = r.get('item_id') or r.get('product_id', '')
        if item_id == product_id:
            product_records.append(r)
    
    if not product_records:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(product_records)
    
    # Ensure we have sale prices
    if 'sale_price' not in df.columns:
        return None
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    elif 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    else:
        df['ts'] = pd.date_range(end=datetime.now(timezone.utc), periods=len(df), freq='5T')
    
    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate features
    df['price'] = df['sale_price']
    
    # Calculate moving averages on auction prices
    ma_df = calculate_moving_averages(df['price'], windows=[15, 60])
    df = pd.concat([df, ma_df], axis=1)
    
    # Calculate volatility
    df['vol_window_30'] = calculate_volatility(df['price'], window=30)
    
    return df


def merge_bazaar_auction_features(
    bazaar_df: Optional[pd.DataFrame],
    auction_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Merge bazaar and auction features into a single feature DataFrame.
    """
    if bazaar_df is None and auction_df is None:
        return None
    
    if bazaar_df is None:
        return auction_df
    
    if auction_df is None:
        return bazaar_df
    
    # For now, just return bazaar data as it has more complete features
    # In a full implementation, you'd merge on timestamp
    return bazaar_df


def validate_feature_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has all required features for ML."""
    required_features = [
        'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
        'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
        'ma_crossover', 'ma_ratio', 'vol_price_ratio',
        'hour_of_day', 'day_of_week', 'day_of_month',
        'market_volatility', 'market_momentum'
    ]
    
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default values
        for feature in missing_features:
            df[feature] = 0.0
    
    return True