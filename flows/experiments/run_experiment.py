from typing import List, Dict
from flows.core.monte_carlo import MonteCarloAnalyzer, MCState
from flows.core.thermodynamics import PersonalityThermodynamics
from pathlib import Path
import json
from datetime import datetime
import logging
import asyncio
from flows.core.llm_client import LLMClient
import numpy as np
import random

class PersonalityPhaseExperiment:
    def __init__(self):
        self.data_dir = Path("data")
        self.generations_dir = self.data_dir / "generations"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories if they don't exist
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.thermodynamics = PersonalityThermodynamics()
        self.llm_client = LLMClient(api_key=None)
        self.mc_analyzer = MonteCarloAnalyzer(
            thermodynamics=self.thermodynamics,
            llm_client=self.llm_client
        )

    async def run_experiment(self, base_personality: Dict, parameters: Dict) -> str:
        """Run phase experiment with multiple personalities
        
        Args:
            base_personality: Base personality configuration
            parameters: Experiment parameters including:
                - n_samples: Total number of temperature samples
                - temp_range: (min_temp, max_temp) tuple
                - prompts: List of prompts to test
                - n_steps: Steps per temperature point
                - batch_size: Batch size for processing
                - n_personalities: Number of personality variations
        """
        # Generate diverse personalities
        n_personalities = parameters.get('n_personalities', 5)
        personalities = generate_diverse_personalities(base_personality, n_personalities)
        
        # Run samples for each personality
        all_states = []
        for personality in personalities:
            states = await self._run_temperature_sample(
                personality=personality,
                temperature=parameters.get('temperature', 0.7),
                prompts=parameters.get('prompts', ["Tell me about yourself"]),
                n_steps=parameters.get('n_steps', 10),
                batch_size=parameters.get('batch_size', 5)
            )
            all_states.extend(states)
        
        # Save results with generation ID
        generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_results(all_states, generation_id)
        
        return generation_id

    def _save_results(self, states: List[MCState], generation_id: str):
        """Save experiment results and metadata"""
        # Save states to JSON
        states_file = self.generations_dir / f"{generation_id}.json"
        states_data = [state.__dict__ for state in states]  # Convert states to dict
        with open(states_file, 'w') as f:
            json.dump(states_data, f, indent=2)
            
        # Save metadata
        metadata_file = self.metadata_dir / f"{generation_id}.json"
        metadata = {
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
            "personality": states[0].personality,  # Get personality from first state
            "parameters": states[0].parameters    # Get parameters from first state
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    async def _run_temperature_sample(
        self,
        temperature: float,
        personality: Dict,
        prompts: List[str],
        n_steps: int = 10,
        batch_size: int = 5
    ) -> List[Dict]:
        """Run a single temperature sample and return serializable states"""
        
        mc_analyzer = MonteCarloAnalyzer(
            thermodynamics=PersonalityThermodynamics(),
            llm_client=self.llm_client
        )
        
        # Get MC states
        states = await mc_analyzer.run_simulation_async(
            initial_personality=personality,
            prompts=prompts,
            n_steps=n_steps,
            batch_size=batch_size,
            temperature=temperature  # Pass temperature explicitly
        )
        
        # Convert MCState objects to serializable dicts, skipping first empty state
        serializable_states = []
        for state in states[1:]:  # Skip first state
            state_dict = {
                "temperature": temperature,  # Use passed temperature
                "energy": state.energy,
                "entropy": state.entropy,
                "enthalpy": state.enthalpy, 
                "coherence": state.coherence,
                "personality": state.personality,
                "phase": state.phase,
                "response": state.response
            }
            serializable_states.append(state_dict)
        
        return serializable_states

def generate_diverse_personalities(base_personality: Dict, n_personalities: int) -> List[Dict]:
    """Generate distinctly different personality profiles for phase separation experiments."""
    personality_archetypes = [
        {
            "I_G": ["Create and explore", "Express emotions", "Inspire others", "Challenge conventions"],
            "I_S": "Creative free spirit",
            "I_W": "World full of possibilities and artistic expression"
        },
        {
            "I_G": ["Analyze and organize", "Maintain order", "Follow procedures", "Achieve goals"],
            "I_S": "Methodical planner",
            "I_W": "Structured environment with clear rules and expectations"
        },
        {
            "I_G": ["Connect with others", "Share experiences", "Build relationships", "Lead groups"],
            "I_S": "Social catalyst",
            "I_W": "Interconnected community focused on collaboration"
        },
        {
            "I_G": ["Analyze problems", "Seek truth", "Question assumptions", "Research deeply"],
            "I_S": "Analytical mind",
            "I_W": "Complex system of interconnected ideas and theories"
        },
        {
            "I_G": ["Support others", "Find harmony", "Mediate conflicts", "Build consensus"],
            "I_S": "Empathetic mediator",
            "I_W": "Balanced environment promoting understanding and growth"
        }
    ]
    
    personalities = []
    for i in range(n_personalities):
        # Select an archetype
        archetype = personality_archetypes[i % len(personality_archetypes)]
        # Create a new personality with the archetype's traits
        personality = {
            "I_G": archetype["I_G"].copy(),
            "I_S": archetype["I_S"],
            "I_W": archetype["I_W"]
        }
        personalities.append(personality)
    
    return personalities

async def run_monte_carlo_experiment():
    """Main entry point for running Monte Carlo experiments"""
    try:
        # Load default configuration
        with open('configs/default_personality.json', 'r') as f:
            personality = json.load(f)
        with open('configs/default_experiment.json', 'r') as f:
            parameters = json.load(f)
            
        # Initialize and run experiment
        experiment = PersonalityPhaseExperiment()
        generation_id = await experiment.run_experiment(personality, parameters)
        
        logging.info(f"Experiment completed successfully. Generation ID: {generation_id}")
        return generation_id
        
    except Exception as e:
        logging.error(f"Error running experiment: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_monte_carlo_experiment())