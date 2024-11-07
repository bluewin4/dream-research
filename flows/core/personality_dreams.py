from typing import Dict, List, Any
import numpy as np
from .personality_sampling import PersonalitySpaceSampling
from .thermodynamics import PersonalityThermodynamics
from .llm_client import LLMClient
from ..personality_generator import PersonalityGenerator
from flows.core.personality_sampling import PersonalityMatrix
from .personality_genetics import PersonalityGenome
from .personality_evolution import PersonalityEvolution

class PersonalityDreams:
    """Manages personality dream states and evolution"""
    
    def __init__(self):
        self.evolution = PersonalityEvolution()
        self.thermo = PersonalityThermodynamics()
        
    async def dream(self, 
                    personality: PersonalityMatrix,
                    duration: int = 10,
                    dream_temp: float = 1.5) -> Dict:
        """Generate and evolve dreams for personality"""
        
        # Initialize dream population
        await self.evolution.initialize_population(personality)
        
        # Define dream fitness function
        async def dream_fitness(genome: PersonalityGenome) -> float:
            coherence = genome.measure_robustness()["trait_coherence"]
            stability = genome.measure_robustness()["global_stability"]
            return (coherence + stability) / 2
        
        # Evolve dreams
        dream_history = await self.evolution.evolve(
            generations=duration,
            fitness_func=dream_fitness,
            target_fitness=0.9
        )
        
        # Select best dream
        best_genome = max(self.evolution.population, 
                         key=lambda g: dream_fitness(g))
        
        return {
            "dream_genome": best_genome,
            "history": dream_history,
            "final_robustness": best_genome.measure_robustness()
        }