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

    async def run_experiment(self, personality: Dict, parameters: Dict) -> str:
        """Run the experiment and return the generation ID"""
        # Generate unique ID for this experiment run
        generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run simulation using MonteCarloAnalyzer
        states = await self.mc_analyzer.run_simulation_async(
            initial_personality=personality,
            prompts=parameters.get('prompts', ["Tell me about yourself"]),
            n_steps=parameters.get('n_steps', 100),
            batch_size=parameters.get('batch_size', 5),
            temperature_schedule=parameters.get('temperature_schedule', [0.1, 0.5, 1.0, 1.5, 2.0])
        )
        
        # Save results
        self._save_results(states, generation_id, personality, parameters)
        
        return generation_id

    def _save_results(self, states: List[MCState], generation_id: str, 
                     personality: Dict, parameters: Dict):
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
            "personality": personality,
            "parameters": parameters
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

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