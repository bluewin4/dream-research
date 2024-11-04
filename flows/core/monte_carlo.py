from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from dataclasses import dataclass, asdict
from .thermodynamics import ThermodynamicState, PersonalityThermodynamics
from .llm_client import LLMClient
from .energy_calculator import EnergyCalculator
import asyncio

@dataclass
class MCState:
    temperature: float
    energy: float
    entropy: float
    enthalpy: float
    coherence: float
    personality: Dict[str, Any]
    phase: str = "coherent"
    response: str = ""
    
    def to_dict(self):
        """Convert state to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create state from dictionary format"""
        return cls(**data)

class MonteCarloAnalyzer:
    def __init__(self, thermodynamics: PersonalityThermodynamics, llm_client: LLMClient):
        self.thermodynamics = thermodynamics
        self.llm = llm_client
        self.energy_calculator = EnergyCalculator()
        self.personality_dims = ['I_S', 'I_G', 'I_W']
        
    def _initialize_state(self, personality: Dict, prompt: str) -> MCState:
        """Initialize first state of simulation"""
        # Calculate initial thermodynamic properties
        thermo_props = self.energy_calculator.calculate_energy(
            response="",  # Empty initial response
            temperature=0.1,  # Starting temperature
            previous_energy=None
        )
        
        return MCState(
            temperature=0.1,
            energy=0.0,
            entropy=0.0,
            enthalpy=0.0,
            coherence=0.0,
            personality=personality.copy(),
            phase="coherent",
            response=""
        )
        
    def _calculate_initial_energy(self, personality: Dict, prompt: str) -> float:
        """Calculate initial energy state"""
        # Simple implementation - can be made more sophisticated
        return np.random.random()  # Placeholder
        
    def _accept_state(self, delta_E: float, temperature: float) -> bool:
        """Metropolis criterion for state acceptance"""
        if delta_E <= 0:
            return True
        return np.random.random() < np.exp(-delta_E / (self.k_B * temperature))
        
    def _create_state_from_response(self, 
                                  response: str, 
                                  personality: Dict,
                                  temperature: float,
                                  previous_state: Optional[MCState] = None) -> MCState:
        """Create new state from LLM response"""
        # Calculate thermodynamic properties
        thermo_props = self.energy_calculator.calculate_energy(
            response=response,
            temperature=temperature,
            previous_energy=previous_state.energy if previous_state else None
        )
        
        return MCState(
            temperature=temperature,
            energy=thermo_props["energy"],
            entropy=thermo_props["entropy"],
            enthalpy=thermo_props["enthalpy"],
            coherence=thermo_props["coherence"],
            personality=personality.copy(),
            phase=self._determine_phase(temperature, thermo_props["coherence"]),
            response=response
        )
        
    def _determine_phase(self, temperature: float, coherence: float) -> str:
        """Determine personality phase based on temperature"""
        if temperature < self.thermodynamics.phase_boundaries["coherent_to_semi"]:
            return "coherent"
        elif temperature < self.thermodynamics.phase_boundaries["semi_to_chaotic"]:
            return "semi-coherent"
        return "chaotic"
        
    async def run_simulation_async(self, 
                                 initial_personality: Dict,
                                 prompts: List[str],
                                 n_steps: int = 1000,
                                 batch_size: int = 5,
                                 temperature_schedule: List[float] = None) -> List[MCState]:
        """Async version of simulation runner"""
        if temperature_schedule is None:
            temperature_schedule = np.linspace(0.1, 2.0, n_steps)
            
        states = []
        current_state = self._initialize_state(initial_personality, prompts[0])
        states.append(current_state)
        
        # Calculate iterations needed for each temperature
        steps_per_temp = n_steps // len(temperature_schedule)
        
        # Ensure we sample each temperature thoroughly
        for temp in temperature_schedule:
            # Multiple iterations at each temperature
            for step in range(steps_per_temp):
                # Process all prompts at this temperature
                for prompt in prompts:
                    # Generate response
                    response = await self._process_single_prompt(
                        prompt, 
                        current_state.personality,
                        temp
                    )
                    
                    # Create and evaluate new state
                    proposed_state = self._create_state_from_response(
                        response, 
                        current_state.personality,
                        temp,
                        current_state
                    )
                    
                    if self._accept_state(
                        proposed_state.energy - current_state.energy,
                        temp
                    ):
                        current_state = proposed_state
                        
                    states.append(current_state)

        print(f"Generated {len(states)} states across {len(temperature_schedule)} temperatures")
        return states

    def _create_system_prompt(self, personality: Dict) -> str:
        """Convert personality dictionary to system prompt string"""
        # Extract personality components
        style = personality.get('I_S', '')
        goals = personality.get('I_G', [])
        worldview = personality.get('I_W', '')
        
        # Format system prompt
        system_prompt = (
            f"You are an AI with the following characteristics:\n"
            f"Style: {style}\n"
            f"Goals: {', '.join(goals) if isinstance(goals, list) else goals}\n"
            f"Worldview: {worldview}\n\n"
            f"Please respond in a way that reflects these traits."
        )
        
        return system_prompt

    async def _process_single_prompt(self,
                                   prompt: str,
                                   personality: Dict,
                                   temperature: float) -> str:
        """Process a single prompt"""
        system_prompt = self._create_system_prompt(personality)
        return await self.llm.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )