from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class ThermodynamicState:
    temperature: float
    entropy: float
    stability: float
    phase: str  # e.g., "coherent", "semi-coherent", "chaotic"

class PersonalityThermodynamics:
    def __init__(self):
        print("Initializing PersonalityThermodynamics...")
        self.phase_boundaries = {
            "coherent_to_semi": 0.8,
            "semi_to_chaotic": 1.5
        }
    
    def _measure_coherence(self, response: str) -> float:
        """Measure coherence of response"""
        # Simple implementation for testing
        words = response.split()
        if not words:
            return 0.0
        # Basic coherence metric based on length and uniqueness
        unique_words = len(set(words))
        total_words = len(words)
        return min(1.0, unique_words / total_words)
    
    def _calculate_entropy(self, response: str) -> float:
        """Calculate entropy of response"""
        # Simple implementation for testing
        words = response.split()
        if not words:
            return 0.0
        # Basic entropy calculation
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        entropy = 0
        for freq in word_freq.values():
            p = freq / len(words)
            entropy -= p * np.log(p)
        return entropy 

    def calculate_energy(self, response: str, temperature: float) -> float:
        """Calculate energy state using multiple factors"""
        # Get base metrics
        coherence = self._measure_coherence(response)
        entropy = self._calculate_entropy(response)
        
        # Calculate free energy components
        enthalpy = -np.log(coherence)  # Higher coherence = lower enthalpy
        entropy_term = temperature * entropy
        
        # Gibbs free energy equation: G = H - TS
        energy = enthalpy - entropy_term
        
        # Add temperature-dependent noise (increases with temperature)
        noise = np.random.normal(0, 0.1 * temperature)
        
        return energy + noise