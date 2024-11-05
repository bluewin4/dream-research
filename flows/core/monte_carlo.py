from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from .thermodynamics import PersonalityThermodynamics
from .types import MCState
from flows.core.llm_client import LLMClient

class MonteCarloAnalyzer:
    def __init__(self, thermodynamics: PersonalityThermodynamics, llm_client: LLMClient):
        self.thermodynamics = thermodynamics
        self.llm = llm_client
        self.k_B = 1.0  # Boltzmann constant
        
    def _initialize_state(self, personality: Dict, prompt: str) -> MCState:
        """Initialize first state of simulation"""
        # Calculate initial thermodynamic properties
        thermo_props = self.thermodynamics.calculate_energy(
            response="",  # Empty initial response
            temperature=0.1,  # Starting temperature
            previous_energy=None
        )
        
        # Convert PersonalityMatrix to dict if it isn't already
        personality_dict = personality.to_dict() if hasattr(personality, 'to_dict') else dict(personality)
        
        return MCState(
            temperature=0.1,
            energy=thermo_props["energy"],
            entropy=thermo_props["entropy"],
            enthalpy=thermo_props["enthalpy"],
            coherence=thermo_props["coherence"],
            personality=personality_dict,  # Use the dictionary version
            phase="coherent",
            response=""
        )
        
    async def run_simulation_async(
        self,
        initial_personality: Dict,
        prompts: List[str],
        n_steps: int,
        batch_size: int,
        temperature: float
    ) -> List[MCState]:
        try:
            states = [self._initialize_state(initial_personality, prompts[0])]
            
            for i in range(n_steps):
                for prompt in prompts:
                    response = await self.llm.generate(
                        prompt=prompt,
                        system_prompt=self._create_system_prompt(initial_personality, temperature),
                        temperature=temperature
                    )
                    
                    # Calculate state properties
                    thermo_props = self.thermodynamics.calculate_energy(
                        response=response,
                        temperature=temperature,
                        previous_energy=states[-1].energy if states else None
                    )
                    
                    state = MCState(
                        temperature=temperature,
                        energy=thermo_props["energy"],
                        entropy=thermo_props["entropy"],
                        enthalpy=thermo_props["enthalpy"],
                        coherence=thermo_props["coherence"],
                        personality=initial_personality,
                        phase=self.thermodynamics._determine_phase(thermo_props["coherence"], temperature),
                        response=response
                    )
                    states.append(state)
                    
            return states
                
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            raise

    def _create_system_prompt(self, personality: Dict, temperature: float) -> str:
        """Creates a system prompt based on personality and temperature.
        
        Args:
            personality: Dictionary containing personality traits
            temperature: Current temperature parameter
            
        Returns:
            Formatted system prompt string
        """
        # Convert personality dict to formatted string
        personality_str = "\n".join([f"- {k}: {v}" for k, v in personality.items()])
        
        # Create base prompt with personality traits
        prompt = f"""Please respond with the following personality traits in mind:
{personality_str}
"""
        return prompt