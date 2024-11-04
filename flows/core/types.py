from dataclasses import dataclass
from typing import Dict

@dataclass
class MCState:
    temperature: float
    energy: float
    entropy: float
    enthalpy: float
    coherence: float
    personality: Dict
    phase: str
    response: str = "" 