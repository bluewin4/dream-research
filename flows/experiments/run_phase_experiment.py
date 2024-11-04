import asyncio
import argparse
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from pathlib import Path
from flows.experiments.run_experiment import PersonalityPhaseExperiment
from flows.core.monte_carlo import MonteCarloAnalyzer, MCState
from flows.core.thermodynamics import PersonalityThermodynamics
from flows.core.personality_matrix import PersonalityMatrix
from flows.core.llm_client import LLMClient
import os
from ..personality_generator import PersonalityGenerator

class PersonalityPhaseExperiment:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.thermodynamics = PersonalityThermodynamics()
        self.llm_client = LLMClient(api_key=os.getenv('LLM_API_KEY'))
        self.personality_generator = PersonalityGenerator(self.thermodynamics)
        
    async def run_experiment(self, parameters: Dict) -> str:
        """Run phase experiment with uniform temperature sampling"""
        # Extract parameters
        n_samples = parameters.get('n_samples', 100)
        temp_range = parameters.get('temp_range', (0.1, 2.0))
        prompts = parameters.get('prompts', ["Tell me about yourself"])
        n_steps = parameters.get('n_steps', 10)
        batch_size = parameters.get('batch_size', 5)
        n_personalities = parameters.get('n_personalities', 5)

        # Generate uniform temperature samples
        temperatures = np.random.uniform(
            low=temp_range[0],
            high=temp_range[1], 
            size=n_samples
        )

        # Generate diverse personalities
        base_temp = (temp_range[0] + temp_range[1]) / 2
        personalities = self.personality_generator.generate_diverse_personalities(
            n_personalities=n_personalities,
            temperature=base_temp
        )

        # Run samples
        all_states = []
        for i, temp in enumerate(temperatures):
            print(f"\nRunning sample {i+1}/{n_samples} at temperature {temp:.2f}")
            
            states = await self._run_temperature_sample(
                personalities=personalities,
                temperature=temp,
                prompts=prompts,
                n_steps=n_steps,
                batch_size=batch_size
            )
            all_states.extend(states)

        # Generate unique ID for this run
        generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        await self._save_results(all_states, generation_id)
        
        return generation_id

    async def _run_temperature_sample(
        self,
        personalities: List[PersonalityMatrix],
        temperature: float,
        prompts: List[str],
        n_steps: int,
        batch_size: int
    ) -> List[MCState]:
        """Run simulation for a single temperature point"""
        all_states = []
        mc_analyzer = MonteCarloAnalyzer(
            thermodynamics=self.thermodynamics,
            llm_client=self.llm_client
        )
        
        for personality in personalities:
            for prompt in prompts:
                states = await mc_analyzer.run_simulation_async(
                    initial_personality=personality,
                    prompts=[prompt],
                    n_steps=n_steps,
                    batch_size=batch_size,
                    temperature_schedule=[temperature]
                )
                all_states.extend(states)
        
        return all_states

    async def _save_results(self, states: List[MCState], generation_id: str):
        """Save states to JSON file with proper structure"""
        output_path = f"data/generations/{generation_id}.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert MCState objects to dictionaries
        serializable_states = [state.to_dict() for state in states]
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_states, f, indent=2)

    def _save_metadata(self, parameters: Dict, generation_id: str):
        """Save experiment metadata"""
        metadata_path = Path(f"data/metadata/{generation_id}.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "parameters": parameters,
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
            "personalities": {
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
    args = parser.parse_args()

    # Load experiment parameters
    with open(args.config) as f:
        parameters = json.load(f)

    # Run experiment
    experiment = PersonalityPhaseExperiment()
    generation_id = await experiment.run_experiment(
        parameters=parameters
    )
    
    print(f"\nExperiment completed! Generation ID: {generation_id}")
    print(f"Data stored in: data/generations/{generation_id}.json")
    print(f"Metadata stored in: data/metadata/{generation_id}.json")

if __name__ == "__main__":
    asyncio.run(main()) 