from typing import Dict, Optional
import numpy as np

class EnergyCalculator:
    """Handles energy calculations for LLM states using thermodynamic principles"""
    
    def __init__(self):
        self.k_B = 1.0  # Boltzmann constant (can be adjusted)
        
    def calculate_energy(self, 
                        response: str, 
                        temperature: float,
                        previous_energy: Optional[float] = None) -> Dict:
        """
        Calculate thermodynamic properties of a response
        
        Args:
            response: The LLM response text
            temperature: Sampling temperature
            previous_energy: Energy of previous state (for delta calculations)
            
        Returns:
            Dict containing:
                - energy: Total Gibbs free energy
                - entropy: Information entropy
                - enthalpy: Calculated enthalpy
                - coherence: Response coherence metric
        """
        # Calculate base metrics
        coherence = self._measure_coherence(response)
        entropy = self._calculate_entropy(response)
        
        # Calculate free energy components
        enthalpy = -np.log(coherence) if coherence > 0 else float('inf')
        entropy_term = temperature * entropy
        
        # Gibbs free energy equation: G = H - TS
        energy = enthalpy - entropy_term
        
        # Add temperature-dependent noise
        noise = np.random.normal(0, 0.1 * temperature)
        total_energy = energy + noise
        
        return {
            "energy": total_energy,
            "entropy": entropy,
            "enthalpy": enthalpy,
            "coherence": coherence,
            "delta_energy": total_energy - previous_energy if previous_energy is not None else 0
        }
    
    def _measure_coherence(self, response: str) -> float:
        """Measure coherence of response"""
        words = response.split()
        if not words:
            return 0.0
        
        # Basic coherence metric based on length and uniqueness
        unique_words = len(set(words))
        total_words = len(words)
        return min(1.0, unique_words / total_words)
    
    def _calculate_entropy(self, response: str) -> float:
        """Calculate information entropy of response"""
        words = response.split()
        if not words:
            return 0.0
        
        # Calculate word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate Shannon entropy
        entropy = 0
        for freq in word_freq.values():
            p = freq / len(words)
            entropy -= p * np.log(p)
        return entropy 