"""
Utility functions for market analysis.
"""

from datetime import datetime, timezone
from typing import Tuple, List


def get_active_sessions(dt: datetime) -> List[str]:
    """
    Get list of active forex sessions for a given datetime.
    
    Args:
        dt: Datetime in UTC
    
    Returns:
        List of active session names
    """
    hour = dt.hour
    active = []
    
    # Sydney: 22:00-07:00 GMT
    if hour >= 22 or hour < 7:
        active.append('sydney')
    
    # Tokyo: 00:00-09:00 GMT
    if hour < 9:
        active.append('tokyo')
    
    # London: 08:00-17:00 GMT
    if 8 <= hour < 17:
        active.append('london')
    
    # New York: 13:00-22:00 GMT
    if 13 <= hour < 22:
        active.append('new_york')
    
    return active


def is_market_open(dt: datetime) -> bool:
    """Check if forex market is open"""
    # Forex is open 24/5
    weekday = dt.weekday()
    return weekday < 5  # Monday-Friday


def calculate_spread(pair: str, session: str = 'london') -> float:
    """
    Estimate typical spread for a currency pair.
    
    Args:
        pair: Currency pair (e.g., 'EURUSD')
        session: Trading session
    
    Returns:
        Spread in pips
    """
    # Typical spreads (in pips)
    base_spreads = {
        'EURUSD': 0.8,
        'GBPUSD': 1.0,
        'USDJPY': 0.9,
        'AUDUSD': 1.2,
    }
    
    spread = base_spreads.get(pair, 2.0)
    
    # Wider spreads during off-hours
    if session in ['sydney', 'tokyo']:
        spread *= 1.5
    
    return spread


def detect_session(hour: int) -> str:
    """
    Detect primary forex session for given hour.
    
    Args:
        hour: Hour in GMT (0-23)
    
    Returns:
        Primary session name
    """
    if 8 <= hour < 13:
        return 'london'
    elif 13 <= hour < 22:
        return 'new_york'
    elif hour < 9:
        return 'tokyo'
    else:
        return 'sydney'
