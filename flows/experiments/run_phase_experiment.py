import asyncio
import argparse
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from pathlib import Path
from flows.experiments.run_experiment import PersonalityPhaseExperiment
from flows.core.monte_carlo import MonteCarloAnalyzer
from flows.core.thermodynamics import PersonalityThermodynamics
from flows.core.llm_client import LLMClient
import os

class PersonalityPhaseExperiment:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_client = LLMClient(api_key=os.getenv('LLM_API_KEY'))
        
    async def run_experiment(self, personality: Dict, parameters: Dict) -> str:
        """
        Run phase experiment with uniform temperature sampling
        
        Args:
            personality: Personality configuration dict
            parameters: Experiment parameters including:
                - n_samples: Total number of temperature samples
                - temp_range: (min_temp, max_temp) tuple
                - prompts: List of prompts to test
                - n_steps: Steps per temperature point
                - batch_size: Batch size for processing
        """
        # Extract parameters
        n_samples = parameters.get('n_samples', 100)
        temp_range = parameters.get('temp_range', (0.1, 2.0))
        prompts = parameters.get('prompts', ["Tell me about yourself"])
        n_steps = parameters.get('n_steps', 10)
        batch_size = parameters.get('batch_size', 5)

        # Generate uniform temperature samples
        temperatures = np.random.uniform(
            low=temp_range[0],
            high=temp_range[1], 
            size=n_samples
        )

        # Run samples
        all_states = []
        for i, temp in enumerate(temperatures):
            print(f"\nRunning sample {i+1}/{n_samples} at temperature {temp:.2f}")
            
            states = await self._run_temperature_sample(
                personality=personality,
                temperature=temp,
                prompts=prompts,
                n_steps=n_steps,
                batch_size=batch_size
            )
            all_states.extend(states)

        # Save results
        generation_id = self.timestamp
        self._save_states(all_states, generation_id)
        self._save_metadata(parameters, personality, generation_id)
        
        return generation_id

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
            batch_size=batch_size
        )
        
        # Convert MCState objects to serializable dicts
        serializable_states = []
        for state in states:
            state_dict = {
                "temperature": state.temperature,
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

    def _save_states(self, states: List[Dict], generation_id: str):
        """Save states to JSON file with proper structure"""
        output_path = f"data/generations/{generation_id}.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write states with proper JSON structure
        with open(output_path, 'w') as f:
            json.dump(states, f, indent=2)

    def _save_metadata(self, parameters: Dict, personality: Dict, generation_id: str):
        """Save experiment metadata"""
        metadata_path = Path(f"data/metadata/{generation_id}.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "parameters": parameters,
            "personality": personality,
            "timestamp": self.timestamp
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _create_initial_state(self):
        return {
            "temperature": 0.1,
            "energy": 0.0,
            "entropy": 0.0,
            "enthalpy": 0.0,
            "coherence": 0.0,
            "personality": {
                "I_G": [
                    "Assist users",
                    "Learn and adapt"
                ],
                "I_S": "Helpful AI assistant",
                "I_W": "Collaborative and supportive environment"
            },
            "phase": "coherent",
            "response": ""
        }

async def main():
    parser = argparse.ArgumentParser(description='Run personality phase experiment')
    parser.add_argument('--config', type=str, default='configs/default_experiment.json',
                       help='Path to experiment configuration file')
    parser.add_argument('--personality', type=str, default='configs/default_personality.json',
                       help='Path to personality configuration file')
    args = parser.parse_args()

    # Load configurations
    with open(args.config) as f:
        parameters = json.load(f)
    with open(args.personality) as f:
        personality = json.load(f)

    # Run experiment
    experiment = PersonalityPhaseExperiment()
    generation_id = await experiment.run_experiment(
        personality=personality,
        parameters=parameters
    )
    
    print(f"\nExperiment completed! Generation ID: {generation_id}")
    print(f"Data stored in: data/generations/{generation_id}.json")
    print(f"Metadata stored in: data/metadata/{generation_id}.json")

if __name__ == "__main__":
    asyncio.run(main()) 