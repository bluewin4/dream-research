from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from scipy import stats
from scipy.stats import entropy

@dataclass
class ThermodynamicParameters:
    """Physical parameters for the thermodynamic calculations"""
    beta: float = 1.0      # Inverse temperature scale
    T_c: float = 1.0       # Critical temperature
    alpha: float = 0.1     # Temperature coupling to enthalpy
    noise_scale: float = 0.1
    epsilon: float = 1e-10 # Numerical stability factor

class PersonalityThermodynamics:
    """Enhanced thermodynamics calculator with improved temperature handling"""
    
    def __init__(self, params: Optional[ThermodynamicParameters] = None):
        print("Initializing PersonalityThermodynamics...")
        self.params = params or ThermodynamicParameters()
        self.phase_boundaries = {
            "coherent_to_semi": 0.8,
            "semi_to_chaotic": 1.5
        }
        
    def calculate_energy(self, 
                        response: str, 
                        temperature: float,
                        previous_energy: Optional[float] = None) -> Dict:
        """
        Calculate thermodynamic properties with enhanced temperature handling
        and phase transition detection
        """
        # Calculate base metrics with improved coherence measure
        coherence = self._measure_coherence(response)
        entropy = self._calculate_entropy(response)
        
        # Calculate order parameter for phase transition detection
        order_param = self._calculate_order_parameter(coherence, temperature)
        
        # Calculate temperature-dependent enthalpy
        enthalpy = self._calculate_enthalpy(coherence, temperature)
        
        # Calculate entropy with improved scaling
        entropy_term = self._calculate_entropy_term(entropy, temperature)
        
        # Calculate free energy with phase transition consideration
        energy = self._calculate_free_energy(enthalpy, entropy_term, order_param, temperature)
        
        # Add sophisticated noise model
        total_energy = self._add_noise(energy, temperature)
        
        result = {
            "energy": total_energy,
            "entropy": entropy,
            "enthalpy": enthalpy,
            "coherence": coherence,
            "order_parameter": order_param,
            "delta_energy": total_energy - previous_energy if previous_energy is not None else 0,
            "phase": self._determine_phase(coherence, temperature),
            "temperature": temperature
        }
        return result

    def _measure_coherence(self, response: str) -> float:
        """Enhanced coherence measurement using multiple metrics"""
        words = response.split()
        if not words:
            return 0.0
            
        # Lexical diversity
        unique_ratio = len(set(words)) / len(words)
        
        # Structural coherence (simple approximation)
        sent_lengths = [len(sent.split()) for sent in response.split('.') if sent.strip()]
        length_variance = np.var(sent_lengths) if sent_lengths else 0
        structural_coherence = 1 / (1 + length_variance)
        
        # Combine metrics
        return 0.7 * unique_ratio + 0.3 * structural_coherence

    def _calculate_entropy(self, response: str) -> float:
        """Calculate information entropy using character and word distributions"""
        if not response:
            return 0.0
            
        # Character-level entropy
        char_freq = np.array([response.count(c) for c in set(response)])
        char_entropy = entropy(char_freq)
        
        # Word-level entropy
        words = response.split()
        word_freq = np.array([words.count(w) for w in set(words)])
        word_entropy = entropy(word_freq) if words else 0
        
        # Combine both entropy measures
        return 0.3 * char_entropy + 0.7 * word_entropy

    def _calculate_order_parameter(self, coherence: float, temperature: float) -> float:
        """Calculate order parameter with critical behavior"""
        T_ratio = temperature / self.params.T_c
        
        if temperature < self.params.T_c:
            # Below T_c: ordered phase
            return np.power(1 - T_ratio, 0.5)  # Mean-field critical exponent
        else:
            # Above T_c: disordered phase with exponential decay
            return np.exp(-T_ratio)

    def _calculate_enthalpy(self, coherence: float, temperature: float) -> float:
        """Calculate temperature-dependent enthalpy"""
        base_enthalpy = -np.log(coherence + self.params.epsilon)
        temp_coupling = 1 + self.params.alpha * temperature
        return base_enthalpy * temp_coupling

    def _calculate_entropy_term(self, entropy: float, temperature: float) -> float:
        """Calculate entropy term with sophisticated temperature scaling"""
        # Sigmoid scaling with critical temperature consideration
        beta_T = self.params.beta * temperature
        scale_factor = 1.0 / (1.0 + np.exp(-beta_T))
        
        # Additional scaling near critical temperature
        T_ratio = temperature / self.params.T_c
        critical_scaling = 1.0 / (1.0 + np.abs(1 - T_ratio))
        
        return scale_factor * critical_scaling * entropy

    def _calculate_free_energy(self, 
                             enthalpy: float, 
                             entropy_term: float,
                             order_param: float, 
                             temperature: float) -> float:
        """Calculate free energy with phase transition effects"""
        # Basic Gibbs free energy
        basic_energy = enthalpy - temperature * entropy_term
        
        # Add phase transition contribution
        phase_contribution = order_param * np.abs(temperature - self.params.T_c)
        
        return basic_energy + phase_contribution

    def _add_noise(self, energy: float, temperature: float) -> float:
        """Add sophisticated noise model"""
        # Temperature-dependent noise scale
        base_scale = self.params.noise_scale * (1.0 - np.exp(-temperature))
        
        # Add critical fluctuations near T_c
        T_diff = np.abs(temperature - self.params.T_c)
        critical_scale = 1.0 + 1.0 / (1.0 + T_diff)
        
        # Combine normal and critical fluctuations
        noise = np.random.normal(0, base_scale) + \
                np.random.normal(0, base_scale * critical_scale * 0.1)
                
        return energy + noise

    def _determine_phase(self, coherence: float, temperature: float) -> str:
        """Determine the phase based on coherence and temperature"""
        if temperature < self.phase_boundaries["coherent_to_semi"]:
            return "coherent"
        elif temperature < self.phase_boundaries["semi_to_chaotic"]:
            return "semi-coherent"
        return "chaotic"

    def validate_energy_landscape(self, states: List[Dict]) -> Dict[str, float]:
        """Validate energy landscape properties"""
        temperatures = [s["temperature"] for s in states]
        energies = [s["energy"] for s in states]
        
        # Calculate correlation
        try:
            energy_temp_correlation = stats.pearsonr(temperatures, energies)[0]
        except:
            energy_temp_correlation = float('nan')
            
        # Analyze phase transitions
        sorted_states = sorted(states, key=lambda x: x["temperature"])
        energies_arr = np.array([s["energy"] for s in sorted_states])
        temps_arr = np.array([s["temperature"] for s in sorted_states])
        d_energy = np.gradient(energies_arr, temps_arr)
        
        return {
            "energy_temp_correlation": energy_temp_correlation,
            "max_energy_derivative": float(np.max(np.abs(d_energy))),
            "transition_temperature": float(temps_arr[np.argmax(np.abs(d_energy))]),
            "transition_sharpness": float(np.std(d_energy))
        }