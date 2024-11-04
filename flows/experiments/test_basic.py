import os
import asyncio
from dotenv import load_dotenv
from flows.core.llm_client import LLMClient
from flows.core.thermodynamics import PersonalityThermodynamics
from flows.core.monte_carlo import MonteCarloAnalyzer
from flows.core.personality_matrix import PersonalityMatrix
from flows.visualization.monte_carlo_viz import MonteCarloVisualizer

async def run_basic_interaction():
    print("Starting basic interaction test...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No OpenAI API key found in .env file")
    else:
        print("OpenAI API key loaded successfully")
    
    try:
        # Initialize components
        print("\nInitializing components...")
        llm = LLMClient(api_key=api_key)
        thermodynamics = PersonalityThermodynamics()
        mc_analyzer = MonteCarloAnalyzer(thermodynamics, llm_client=llm)
        print("Components initialized successfully")
        
        # Create test personality
        print("\nCreating test personality...")
        personality = {
            'I_S': 'analytical logical systematic',
            'I_G': ['understand', 'solve', 'optimize'],
            'I_W': 'structured rational world'
        }
        personality_matrix = PersonalityMatrix.from_dict(personality)
        print(f"Personality created: {personality_matrix.to_dict()}")
        
        # Test prompt
        prompt = "Analyze the relationship between temperature and stability"
        print(f"\nUsing test prompt: {prompt}")
        
        # Generate initial response using LLM
        print("\nGenerating initial LLM response...")
        response = await llm.generate_async(
            prompt=prompt,
            system_prompt="You are a scientific analyzer focusing on thermodynamic systems.",
            temperature=0.7
        )
        print(f"Initial response received: {response[:100]}...")
        
        # Run Monte Carlo simulation
        print("\nRunning Monte Carlo simulation...")
        n_temperatures = 10  # Number of temperature points
        repeats = 10  # Number of repeats per temperature
        
        all_states = []
        for i in range(n_temperatures):
            print(f"\nRunning temperature point {i+1}/{n_temperatures}")
            for j in range(repeats):
                states = await mc_analyzer.run_simulation_async(
                    initial_personality=personality,
                    prompts=[prompt],
                    n_steps=10,
                    batch_size=5
                )
                all_states.extend(states)
                print(f"  Completed repeat {j+1}/{repeats}")
        
        print(f"\nSimulation completed with {len(all_states)} total states")
        
        # Basic analysis of results
        if all_states:
            print("\nAnalyzing results...")
            temperatures = [state.temperature for state in all_states]
            energies = [state.energy for state in all_states]
            
            # Calculate average energy per temperature point
            temp_energy_map = {}
            for state in all_states:
                if state.temperature not in temp_energy_map:
                    temp_energy_map[state.temperature] = []
                temp_energy_map[state.temperature].append(state.energy)
            
            avg_energies = {temp: sum(energies)/len(energies) 
                          for temp, energies in temp_energy_map.items()}
            
            print(f"Temperature range: {min(temperatures):.2f} to {max(temperatures):.2f}")
            print(f"Energy range: {min(energies):.2f} to {max(energies):.2f}")
            print(f"Number of unique temperature points: {len(temp_energy_map)}")
            print(f"Average number of samples per temperature: {len(all_states)/len(temp_energy_map):.1f}")
            
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
    finally:
        print("\nTest completed")

def test_basic_interaction():
    """Wrapper function to run async test"""
    asyncio.run(run_basic_interaction())

if __name__ == "__main__":
    test_basic_interaction() 