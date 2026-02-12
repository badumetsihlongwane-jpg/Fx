"""Model components for NEXUS-FX"""

from .nexus_fx import NEXUSFX
from .associative_memory import AssociativeMemory
from .self_modifying_titans import SelfModifyingTitans
from .continuum_memory import ContinuumMemorySystem
from .cross_pair_memory import CrossPairMemory
from .session_gate import SessionFrequencyGate
from .regime_detector import RegimeDetector
from .output_heads import OutputHeads

__all__ = [
    "NEXUSFX",
    "AssociativeMemory",
    "SelfModifyingTitans",
    "ContinuumMemorySystem",
    "CrossPairMemory",
    "SessionFrequencyGate",
    "RegimeDetector",
    "OutputHeads",
]
