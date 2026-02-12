"""
Session Clock - Forex session detection and timing features.

Detects active trading sessions and computes session-related features:
- Sydney (22:00-07:00 GMT)
- Tokyo (00:00-09:00 GMT)
- London (08:00-17:00 GMT)
- New York (13:00-22:00 GMT)

Also tracks session overlaps which are high-volatility periods.
"""

import torch
import numpy as np
from datetime import datetime, timezone
from typing import Tuple


class SessionClock:
    """
    Forex session detection and timing features.
    
    Generates features based on active trading sessions and their characteristics.
    """
    
    def __init__(self):
        # Session hours in GMT (24-hour format)
        self.sessions = {
            'sydney': (22, 7),    # 22:00-07:00 GMT
            'tokyo': (0, 9),      # 00:00-09:00 GMT
            'london': (8, 17),    # 08:00-17:00 GMT
            'new_york': (13, 22), # 13:00-22:00 GMT
        }
        
        # Session names for indexing
        self.session_names = ['sydney', 'tokyo', 'london', 'new_york']
    
    def detect_sessions(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Detect active sessions for given timestamps.
        
        Args:
            timestamps: Unix timestamps (batch, seq_len)
        
        Returns:
            session_indicators: Binary indicators (batch, seq_len, 6)
                [is_sydney, is_tokyo, is_london, is_ny, is_overlap, is_weekend]
        """
        batch_size, seq_len = timestamps.shape
        indicators = torch.zeros(batch_size, seq_len, 6)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                hour = dt.hour
                weekday = dt.weekday()  # 0=Monday, 6=Sunday
                
                # Check each session
                is_sydney = self._is_in_session(hour, *self.sessions['sydney'])
                is_tokyo = self._is_in_session(hour, *self.sessions['tokyo'])
                is_london = self._is_in_session(hour, *self.sessions['london'])
                is_ny = self._is_in_session(hour, *self.sessions['new_york'])
                
                # Overlap detection (multiple sessions active)
                num_active = sum([is_sydney, is_tokyo, is_london, is_ny])
                is_overlap = float(num_active > 1)
                
                # Weekend detection
                is_weekend = float(weekday >= 5)  # Saturday or Sunday
                
                indicators[b, t, 0] = float(is_sydney)
                indicators[b, t, 1] = float(is_tokyo)
                indicators[b, t, 2] = float(is_london)
                indicators[b, t, 3] = float(is_ny)
                indicators[b, t, 4] = is_overlap
                indicators[b, t, 5] = is_weekend
        
        return indicators
    
    def _is_in_session(self, hour: int, start: int, end: int) -> bool:
        """Check if hour is within session"""
        if start < end:
            return start <= hour < end
        else:
            # Session crosses midnight (e.g., Sydney)
            return hour >= start or hour < end
    
    def compute_session_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute rich session features.
        
        Args:
            timestamps: Unix timestamps (batch, seq_len)
        
        Returns:
            features: Session features (batch, seq_len, feature_dim)
                - Session indicators (6 binary)
                - Time to session open/close (2 continuous)
                - Session volatility profile (4 continuous, one per session)
                - Day of week encoding (7 one-hot)
        """
        batch_size, seq_len = timestamps.shape
        
        # Session indicators
        session_indicators = self.detect_sessions(timestamps)
        
        # Time to next session change
        time_features = self._compute_time_features(timestamps)
        
        # Session volatility profiles (known characteristics)
        vol_profiles = self._get_session_volatility_profiles(session_indicators)
        
        # Day of week
        dow_features = self._encode_day_of_week(timestamps)
        
        # Concatenate all features
        features = torch.cat([
            session_indicators,  # 6
            time_features,       # 2
            vol_profiles,        # 4
            dow_features,        # 7
        ], dim=-1)
        
        return features
    
    def _compute_time_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time-to-event features.
        
        Returns:
            time_features: (batch, seq_len, 2)
                - Hours to next session open
                - Hours to next session close
        """
        batch_size, seq_len = timestamps.shape
        time_features = torch.zeros(batch_size, seq_len, 2)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                hour = dt.hour
                minute = dt.minute
                
                # Find next session transition
                # Simplified: distance to London open (most important) and NY close
                hours_to_london = (8 - hour) % 24
                hours_to_ny_close = (22 - hour) % 24
                
                time_features[b, t, 0] = hours_to_london + minute / 60
                time_features[b, t, 1] = hours_to_ny_close + minute / 60
        
        return time_features
    
    def _get_session_volatility_profiles(self, session_indicators: torch.Tensor) -> torch.Tensor:
        """
        Encode known volatility characteristics of each session.
        
        Historical volatility patterns:
        - Sydney: Low (0.3)
        - Tokyo: Medium-Low (0.5)
        - London: High (0.9)
        - New York: Very High (1.0)
        """
        batch_size, seq_len, _ = session_indicators.shape
        vol_profiles = torch.zeros(batch_size, seq_len, 4)
        
        # Volatility weights
        vol_weights = torch.tensor([0.3, 0.5, 0.9, 1.0])
        
        # Apply to active sessions
        vol_profiles = session_indicators[:, :, :4] * vol_weights.unsqueeze(0).unsqueeze(0)
        
        return vol_profiles
    
    def _encode_day_of_week(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        One-hot encoding of day of week.
        
        Monday=0, ..., Sunday=6
        """
        batch_size, seq_len = timestamps.shape
        dow_features = torch.zeros(batch_size, seq_len, 7)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                weekday = dt.weekday()
                dow_features[b, t, weekday] = 1.0
        
        return dow_features
    
    def get_session_embedding(self, session_indicators: torch.Tensor) -> torch.Tensor:
        """
        Convert session indicators to learned embedding.
        
        This is a simple weighted sum; in practice, use a learned embedding layer.
        
        Args:
            session_indicators: (batch, seq_len, 6)
        
        Returns:
            session_embedding: (batch, seq_len, embedding_dim)
        """
        # Simple weighted combination as a placeholder
        # In the full model, this would be a learned embedding
        batch_size, seq_len, _ = session_indicators.shape
        
        # Weight matrix (6 sessions → 32 dim embedding)
        # This is a simplified version; use nn.Linear in practice
        embedding_dim = 32
        weights = torch.randn(6, embedding_dim) * 0.1
        
        session_embedding = torch.matmul(session_indicators, weights)
        
        return session_embedding
