"""
NEXUS-FX: Nested EXchange Understanding System for Forex

A hierarchical forex trading architecture based on nested associative memories
operating at different timescales, inspired by the HOPE/NSAM framework.

Core principle: Treat the entire model as a hierarchy of nested optimization
problems with associative memories at different update frequencies matching
market dynamics (tick-level microstructure to macro regime shifts).
"""

__version__ = "0.1.0"
__author__ = "NEXUS-FX Team"

from .config import NexusFXConfig

__all__ = ["NexusFXConfig"]
