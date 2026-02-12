"""
Macro Feature Encoding - Economic calendar, rates, yields, commodities.

Encodes macro-fundamental data that affects forex markets:
- Economic calendar events (NFP, CPI, rate decisions)
- Interest rates (Fed, ECB, BoJ, RBA, BoE)
- Bond yields (US10Y, EU10Y, JP10Y, AU10Y)
- Commodities (Gold, Oil, DXY)
- Sentiment proxies (VIX, risk-on/risk-off)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MacroFeatureEncoder:
    """
    Encodes macro-fundamental features for forex trading.
    
    Works with limited data availability:
    - Economic calendar: time-to-event, expected/actual/previous values
    - Interest rates: current rates and differentials
    - Bond yields: current yields and spreads
    - Commodities: current prices
    - Sentiment: current VIX level, risk-on/off classification
    
    Args:
        pairs: List of currency pairs
        include_calendar: Whether to include economic calendar events
        include_rates: Whether to include interest rates
        include_yields: Whether to include bond yields
        include_commodities: Whether to include commodity prices
        include_sentiment: Whether to include sentiment indicators
    """
    
    def __init__(
        self,
        pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        include_calendar: bool = True,
        include_rates: bool = True,
        include_yields: bool = True,
        include_commodities: bool = True,
        include_sentiment: bool = True,
    ):
        self.pairs = pairs
        self.include_calendar = include_calendar
        self.include_rates = include_rates
        self.include_yields = include_yields
        self.include_commodities = include_commodities
        self.include_sentiment = include_sentiment
        
        # Extract currencies from pairs
        self.currencies = self._extract_currencies()
        
        # Feature dimension
        self.feature_dim = self._calculate_feature_dim()
    
    def _extract_currencies(self) -> set:
        """Extract unique currencies from pairs"""
        currencies = set()
        for pair in self.pairs:
            if len(pair) == 6:
                currencies.add(pair[:3])
                currencies.add(pair[3:])
        return currencies
    
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        dim = 0
        
        if self.include_calendar:
            dim += 10  # Calendar event encoding
        
        if self.include_rates:
            dim += len(self.currencies) + len(self.pairs)  # Rates + differentials
        
        if self.include_yields:
            dim += len(self.currencies) + len(self.pairs)  # Yields + spreads
        
        if self.include_commodities:
            dim += 3  # Gold, Oil, DXY
        
        if self.include_sentiment:
            dim += 2  # VIX level, risk-on/off
        
        return dim
    
    def encode(
        self,
        timestamps: torch.Tensor,
        calendar_data: Optional[pd.DataFrame] = None,
        rates_data: Optional[Dict[str, float]] = None,
        yields_data: Optional[Dict[str, float]] = None,
        commodities_data: Optional[Dict[str, float]] = None,
        sentiment_data: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Encode macro features for given timestamps.
        
        Args:
            timestamps: Timestamps to encode for (batch, seq_len)
            calendar_data: DataFrame with economic events
            rates_data: Dict of current interest rates by currency
            yields_data: Dict of current bond yields by currency
            commodities_data: Dict of commodity prices
            sentiment_data: Dict of sentiment indicators
        
        Returns:
            macro_features: Encoded features (batch, seq_len, feature_dim)
        """
        batch_size, seq_len = timestamps.shape
        features = []
        
        # 1. Economic Calendar
        if self.include_calendar:
            calendar_features = self._encode_calendar(timestamps, calendar_data)
            features.append(calendar_features)
        
        # 2. Interest Rates
        if self.include_rates:
            rates_features = self._encode_rates(timestamps, rates_data)
            features.append(rates_features)
        
        # 3. Bond Yields
        if self.include_yields:
            yields_features = self._encode_yields(timestamps, yields_data)
            features.append(yields_features)
        
        # 4. Commodities
        if self.include_commodities:
            commodity_features = self._encode_commodities(timestamps, commodities_data)
            features.append(commodity_features)
        
        # 5. Sentiment
        if self.include_sentiment:
            sentiment_features = self._encode_sentiment(timestamps, sentiment_data)
            features.append(sentiment_features)
        
        # Concatenate all features
        if features:
            macro_features = torch.cat(features, dim=-1)
        else:
            macro_features = torch.zeros(batch_size, seq_len, 1)
        
        return macro_features
    
    def _encode_calendar(
        self,
        timestamps: torch.Tensor,
        calendar_data: Optional[pd.DataFrame],
    ) -> torch.Tensor:
        """
        Encode economic calendar events.
        
        Features:
        - Time to next major event (hours)
        - Event importance (0-3: low/medium/high/critical)
        - Expected impact direction (- to +)
        - Surprise factor (actual - expected, normalized)
        - Event type encoding (one-hot for NFP/CPI/Rate/GDP/Other)
        """
        batch_size, seq_len = timestamps.shape
        
        # Default: no events
        features = torch.zeros(batch_size, seq_len, 10)
        
        if calendar_data is not None and len(calendar_data) > 0:
            # Convert timestamps to datetime
            for b in range(batch_size):
                for t in range(seq_len):
                    ts = timestamps[b, t].item()
                    dt = datetime.fromtimestamp(ts)
                    
                    # Find next event
                    future_events = calendar_data[calendar_data['timestamp'] > dt]
                    if len(future_events) > 0:
                        next_event = future_events.iloc[0]
                        time_to_event = (next_event['timestamp'] - dt).total_seconds() / 3600
                        
                        features[b, t, 0] = min(time_to_event / 24, 10)  # Days to event, capped at 10
                        features[b, t, 1] = next_event.get('importance', 1) / 3  # Normalized
                        features[b, t, 2] = next_event.get('expected_direction', 0)
                        features[b, t, 3] = next_event.get('surprise', 0)
                        
                        # Event type one-hot
                        event_type = next_event.get('type', 'Other')
                        type_idx = {'NFP': 4, 'CPI': 5, 'Rate': 6, 'GDP': 7, 'Other': 8}.get(event_type, 8)
                        features[b, t, type_idx] = 1.0
        
        return features
    
    def _encode_rates(
        self,
        timestamps: torch.Tensor,
        rates_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode interest rates and differentials.
        
        Features:
        - Current rate for each currency
        - Rate differential for each pair
        """
        batch_size, seq_len = timestamps.shape
        
        # Map currencies to rates
        currency_map = {
            'USD': 'FED',
            'EUR': 'ECB',
            'GBP': 'BOE',
            'JPY': 'BOJ',
            'AUD': 'RBA',
        }
        
        num_currencies = len(self.currencies)
        num_pairs = len(self.pairs)
        
        features = torch.zeros(batch_size, seq_len, num_currencies + num_pairs)
        
        if rates_data is not None:
            # Currency rates
            for i, currency in enumerate(sorted(self.currencies)):
                rate_key = currency_map.get(currency, currency)
                rate = rates_data.get(rate_key, 0.0)
                features[:, :, i] = rate
            
            # Pair differentials
            for i, pair in enumerate(self.pairs):
                if len(pair) == 6:
                    base_curr = pair[:3]
                    quote_curr = pair[3:]
                    
                    base_rate = rates_data.get(currency_map.get(base_curr, base_curr), 0.0)
                    quote_rate = rates_data.get(currency_map.get(quote_curr, quote_curr), 0.0)
                    
                    differential = base_rate - quote_rate
                    features[:, :, num_currencies + i] = differential
        
        return features
    
    def _encode_yields(
        self,
        timestamps: torch.Tensor,
        yields_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode bond yields and spreads.
        
        Similar to rates encoding.
        """
        batch_size, seq_len = timestamps.shape
        
        yield_map = {
            'USD': 'US10Y',
            'EUR': 'EU10Y',
            'GBP': 'UK10Y',
            'JPY': 'JP10Y',
            'AUD': 'AU10Y',
        }
        
        num_currencies = len(self.currencies)
        num_pairs = len(self.pairs)
        
        features = torch.zeros(batch_size, seq_len, num_currencies + num_pairs)
        
        if yields_data is not None:
            # Currency yields
            for i, currency in enumerate(sorted(self.currencies)):
                yield_key = yield_map.get(currency, currency)
                yield_val = yields_data.get(yield_key, 0.0)
                features[:, :, i] = yield_val
            
            # Pair spreads
            for i, pair in enumerate(self.pairs):
                if len(pair) == 6:
                    base_curr = pair[:3]
                    quote_curr = pair[3:]
                    
                    base_yield = yields_data.get(yield_map.get(base_curr, base_curr), 0.0)
                    quote_yield = yields_data.get(yield_map.get(quote_curr, quote_curr), 0.0)
                    
                    spread = base_yield - quote_yield
                    features[:, :, num_currencies + i] = spread
        
        return features
    
    def _encode_commodities(
        self,
        timestamps: torch.Tensor,
        commodities_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode commodity prices.
        
        Features:
        - Gold (safe haven)
        - Oil (WTI or Brent)
        - DXY (US Dollar Index)
        """
        batch_size, seq_len = timestamps.shape
        features = torch.zeros(batch_size, seq_len, 3)
        
        if commodities_data is not None:
            features[:, :, 0] = commodities_data.get('Gold', 0.0) / 2000  # Normalize
            features[:, :, 1] = commodities_data.get('Oil', 0.0) / 100    # Normalize
            features[:, :, 2] = commodities_data.get('DXY', 0.0) / 100    # Normalize
        
        return features
    
    def _encode_sentiment(
        self,
        timestamps: torch.Tensor,
        sentiment_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode sentiment indicators.
        
        Features:
        - VIX level (volatility index)
        - Risk-on/risk-off classification
        """
        batch_size, seq_len = timestamps.shape
        features = torch.zeros(batch_size, seq_len, 2)
        
        if sentiment_data is not None:
            vix = sentiment_data.get('VIX', 15.0)
            features[:, :, 0] = vix / 50  # Normalize
            
            # Risk-on/off: -1 (risk-off) to +1 (risk-on)
            risk_sentiment = sentiment_data.get('risk_sentiment', 0.0)
            features[:, :, 1] = risk_sentiment
        
        return features
