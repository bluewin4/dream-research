from typing import Dict, List
from flows.core.personality_dreams import PersonalityDreams
from flows.core.llm_client import LLMClient
import os
from dotenv import load_dotenv
import asyncio

async def run_dream_evolution_experiment():
    # Load environment variables from .env file
    load_dotenv()

    # Create LLMClient first
    llm = LLMClient()

    # Initialize with base personality and trait pools
    base_personality = {
        "I_G": ["Assist users", "Learn and adapt"],
        "I_S": "Helpful AI assistant",
        "I_W": "Collaborative and supportive environment"
    }

    trait_pools = {
        "goals": [
            ["Analyze problems", "Generate solutions"],
            ["Learn and adapt", "Optimize processes"],
            ["Assist users", "Share knowledge"]
        ],
        "self_image": [
            "Adaptive problem-solving system",
            "Helpful AI assistant",
            "Knowledge sharing entity"
        ],
        "worldview": [
            "Dynamic knowledge-driven environment",
            "Collaborative and supportive environment",
            "Growth-oriented learning space"
        ]
    }

    # Create PersonalityDreams instance with llm
    dreams = PersonalityDreams(
        base_temperature=0.7,
        max_temperature=2.0,
        base_personality=base_personality,
        trait_pools=trait_pools,
        llm=llm
    )

    # Generate dream sequence
    sequence = await dreams.generate_dream_sequence(
        personality=base_personality,
        prompt="Tell me about your role in helping users.",
        steps=5
    )

    # Interpret the dreams
    interpretation = await dreams.interpret_dream(sequence, base_personality)
    
    return {
        'sequence': sequence,
        'interpretation': interpretation
    }

async def main():
    print("Initializing PersonalityThermodynamics...")
    results = await run_dream_evolution_experiment()
    
    print("Dream Sequence:")
    for i, dream in enumerate(results['sequence']):
        print(f"\nDream {i+1}:")
        print(f"Temperature: {dream['temperature']}")
        print(f"Phase: {dream['phase']}")
        print(f"Coherence: {dream['coherence']}")
        print(f"Response: {dream['response'][:100]}...")
        
    print("\nInterpretation:")
    for key, value in results['interpretation'].items():
        print(f"\n{key.capitalize()}:")
        print(value[:200]) 

if __name__ == "__main__":
    asyncio.run(main())