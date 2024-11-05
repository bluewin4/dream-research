from dotenv import load_dotenv
import asyncio
import argparse
import numpy as np
from typing import Dict, List, Any,  Optional
import json
from datetime import datetime
from pathlib import Path
from flows.core.monte_carlo import MonteCarloAnalyzer
from flows.core.thermodynamics import PersonalityThermodynamics
from flows.core.personality_matrix import PersonalityMatrix
from flows.core.llm_client import LLMClient
import os

from flows.core.types import MCState
from ..personality_generator import PersonalityGenerator
import aiofiles
import time
from flows.core.personality_dreams import PersonalityDreams

# Add this at the start of your script
load_dotenv()

class PersonalityPhaseExperiment:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.thermodynamics = PersonalityThermodynamics()
        self.llm_client = llm_client or LLMClient()
        self.monte_carlo = MonteCarloAnalyzer(self.thermodynamics, self.llm_client)
        self.generations_dir = Path("data/generations")
        self.metadata_dir = Path("data/metadata")
        
        # Ensure directories exist
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.personality_generator = PersonalityGenerator()

    async def run_experiment(self, parameters: Dict) -> str:
        """Run phase transition experiment with improved error handling"""
        print("Starting experiment with parameters:", parameters)
        
        # Generate random temperatures within the range
        temp_range = parameters.get('temp_range', [0.1, 2.0])
        n_steps = parameters.get('n_steps', 10)
        temperatures = np.random.uniform(low=temp_range[0], high=temp_range[1], size=n_steps)
        
        print(f"Running experiment across {n_steps} temperature points")
        
        all_states = []
        for i, temp in enumerate(temperatures):
            print(f"Processing temperature point {i+1}/{n_steps}: T={temp:.2f}")
            states = await self._run_temperature_sample(
                temperature=temp,
                prompts=parameters.get('prompts', ["Tell me about yourself"]),
                n_steps=parameters.get('n_steps', 10),
                batch_size=parameters.get('batch_size', 5)
            )
            
            if states:
                print(f"Generated {len(states)} states for temperature {temp:.2f}")
                all_states.extend(states)
            else:
                print(f"Warning: No states generated for temperature {temp:.2f}")
            
        if not all_states:
            raise Exception("No valid states generated across all temperatures")
            
        # Save results
        generation_id = f"phase_exp_{int(datetime.now().timestamp())}"
        print(f"Saving {len(all_states)} total states with ID: {generation_id}")
        await self._save_results(all_states, generation_id, parameters)
        return generation_id

    async def _save_results(self, states: List[Dict], generation_id: str, parameters: Dict):
        """Save experiment results with metadata"""
        # Convert MCState objects to dictionaries
        serialized_states = []
        for state in states:
            # Check if personality is already a dict or needs conversion
            personality_dict = state.personality if isinstance(state.personality, dict) else state.personality.to_dict()
            
            state_dict = {
                "temperature": state.temperature,
                "energy": state.energy,
                "entropy": state.entropy,
                "enthalpy": state.enthalpy,
                "coherence": state.coherence,
                "personality": personality_dict,
                "phase": state.phase,
                "response": state.response
            }
            serialized_states.append(state_dict)

        output = {
            "metadata": {
                "experiment_id": generation_id,
                "timestamp": datetime.now().isoformat(),
                "parameters": parameters,
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "OpenAI",
                    "parameters": {
                        "max_tokens": 100,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "response_format": {"type": "text"},
                        "seed": None
                    }
                }
            },
            "states": serialized_states
        }
        
        output_file = self.generations_dir / f"{generation_id}.json"
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(output, indent=2))

    async def _run_temperature_sample(self,
                                    temperature: float,
                                    prompts: List[str],
                                    n_steps: int,
                                    batch_size: int) -> List[MCState]:
        """Run Monte Carlo sampling at a specific temperature"""
        try:
            # Generate multiple personalities for the batch
            states = []
            for _ in range(batch_size):
                # Generate a new personality for each sample
                personality = self.personality_generator.generate()
                
                # Run Monte Carlo simulation for each personality
                batch_states = await self.monte_carlo.run_simulation_async(
                    initial_personality=personality,
                    prompts=prompts,
                    n_steps=1,  # Changed to 1 since we're handling batching here
                    batch_size=1,
                    temperature=temperature
                )
                
                valid_states = [
                    state for state in batch_states 
                    if state.response and not state.response.startswith("Error:")
                ]
                
                if valid_states:
                    states.extend(valid_states)
            
            if not states:
                print(f"Warning: No valid states generated for temperature {temperature}")
                fallback_state = MCState(
                    temperature=temperature,
                    energy=0.0,
                    entropy=0.0,
                    enthalpy=0.0,
                    coherence=0.0,
                    personality=self.personality_generator.generate(),
                    phase="unknown",
                    response="Error: Failed to generate valid response"
                )
                return [fallback_state]
                    
            return states
                
        except Exception as e:
            print(f"Error in temperature sample {temperature}: {str(e)}")
            return [MCState(
                temperature=temperature,
                energy=0.0,
                entropy=0.0,
                enthalpy=0.0,
                coherence=0.0,
                personality=self.personality_generator.generate(),
                phase="error",
                response=f"Error: {str(e)}"
            )]

def load_parameters(config_path: str = "configs/default_experiment.json") -> Dict[str, Any]:
    """
    Load experiment parameters from config file
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Dictionary containing experiment parameters
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Extract experiment parameters
        parameters = {
            "n_samples": config["experiment"]["n_samples"],
            "temp_range": config["experiment"]["temp_range"],
            "n_steps": config["experiment"]["n_steps"],
            "batch_size": config["experiment"]["batch_size"],
            "prompts": config["experiment"]["prompts"],
            "model": config["model"]
        }
        
        return parameters
        
    except Exception as e:
        print(f"Error loading parameters: {str(e)}")
        # Provide fallback default parameters
        return {
            "n_samples": 3,
            "temp_range": [0.1, 2.0],
            "n_steps": 2,
            "batch_size": 1,
            "prompts": ["Tell me about yourself"],
            "model": {
                "name": "gpt-4",
                "max_tokens": 100,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "response_format": {"type": "text"}
            }
        }

async def main():
    try:
        experiment = PersonalityPhaseExperiment()
        parameters = load_parameters()
        generation_id = await experiment.run_experiment(parameters)
        print(f"Experiment completed. Generation ID: {generation_id}")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 