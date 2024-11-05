from typing import Dict, List
from flows.core.personality_dreams import PersonalityDreams
from flows.core.llm_client import LLMClient
import os
from dotenv import load_dotenv
import asyncio
from flows.experiments.run_phase_experiment import PersonalityPhaseExperiment
import json

async def run_dream_evolution_experiment():
    # Load environment variables from .env file
    load_dotenv()

    # Create LLMClient with model information
    llm = LLMClient(
        model_name="gpt-4o-mini",
        model_version="",
        temperature=0.7,
        max_tokens=1000
    )

    # Initialize experiment with LLM
    experiment = PersonalityPhaseExperiment(llm=llm)
    
    # Run experiment with expanded parameters
    parameters = {
        "n_samples": 3,
        "temp_range": [0.1, 2.0],
        "n_steps": 10,
        "batch_size": 5,
        "prompts": ["Tell me about yourself"]
    }
    
    generation_id = await experiment.run_experiment(parameters)
    
    # Load and return results
    results_file = experiment.generations_dir / f"{generation_id}.json"
    with open(results_file) as f:
        results = json.load(f)
    return results

async def main():
    print("Initializing PersonalityThermodynamics...")
    results = await run_dream_evolution_experiment()
    
    print("\nExperiment Results:")
    print(f"Generated {len(results['states'])} states across {len(set(s['temperature'] for s in results['states']))} temperature points")
    
    # Print sample of results
    print("\nSample States:")
    for i, state in enumerate(results['states'][:3]):
        print(f"\nState {i+1}:")
        print(f"Temperature: {state['temperature']:.2f}")
        print(f"Phase: {state['phase']}")
        print(f"Coherence: {state['coherence']:.2f}")
        print(f"Response: {state['response'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())