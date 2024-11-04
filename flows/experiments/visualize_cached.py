import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from flows.visualization.monte_carlo_viz import MonteCarloVisualizer
from flows.core.monte_carlo import MCState
from flows.core.personality_matrix import PersonalityMatrix
from typing import List
import random

class DataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def save_states(self, states: list, experiment_name: str = None):
        """Save states to JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
        
        # Convert states to serializable format
        serialized_states = [
            {
                'temperature': state.temperature,
                'energy': state.energy,
                'personality': state.personality,
                'phase': state.phase
            }
            for state in states
        ]
        
        filepath = self.data_dir / f"states_{name}.json"
        with open(filepath, 'w') as f:
            json.dump(serialized_states, f, indent=2)
        
        print(f"Saved states to {filepath}")
        
    def load_states(self, filename: str) -> list:
        """Load states from JSON file"""
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert back to MCState objects
        states = [
            MCState(
                temperature=s['temperature'],
                energy=s['energy'],
                personality=s['personality'],
                phase=s['phase']
            )
            for s in data
        ]
        return states

def create_mock_states(response: str, temperature: float) -> List[MCState]:
    """Create mock states with actual temperature from cache key"""
    states = []
    n_samples = 1  # Significantly increased for comprehensive analysis
    
    # Calculate base energy from full response length
    base_energy = -1.0 + 0.5 * temperature - (len(response) / 10000)
    
    for _ in range(n_samples):
        # Add more varied noise to energy calculation
        energy = base_energy + random.gauss(0, 0.2)  # Using Gaussian distribution for more natural variation
        
        # Determine phase based on temperature and response length
        if temperature < 0.3 and len(response) < 500:
            phase = "coherent"
        elif temperature < 0.7 or len(response) < 1000:
            phase = "transitional"
        else:
            phase = "chaotic"
            
        state = MCState(
            temperature=temperature,
            energy=energy,
            personality={
                "I_S": "analytical logical systematic",
                "I_G": ["understand", "solve", "optimize"],
                "I_W": "structured rational world"
            },
            phase=phase,
            response=response
        )
        states.append(state)
    
    return states

def save_plots(viz: MonteCarloVisualizer, experiment_name: str):
    """Save all plots to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Save each figure
    for i in plt.get_fignums():
        fig = plt.figure(i)
        filename = f"{experiment_name}_plot_{i}_{timestamp}.png"
        filepath = plots_dir / filename
        fig.savefig(filepath)
        print(f"Saved plot to {filepath}")

def main():
    # Ensure output directories exist
    Path("data").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    # Read generations instead of cache
    generations_dir = Path('data/generations')
    all_states = []
    
    for file_path in generations_dir.glob('*.json'):
        with open(file_path, 'r') as f:
            generations = json.load(f)
            for gen in generations:
                temperature = float(gen['temperature'])
                states = create_mock_states(gen['response'], temperature)
                all_states.extend(states)
                print(f"Processed entry with temperature: {temperature}")
    
    print(f"Analysis complete. Generated {len(all_states)} total states")
    
    # Sort states by temperature for better visualization
    all_states.sort(key=lambda x: x.temperature)
    
    # Save states
    states_data = [state.to_dict() for state in all_states]
    with open('data/mock_states.json', 'w') as f:
        json.dump(states_data, f, indent=2)
    
    # Initialize visualizer
    viz = MonteCarloVisualizer()
    
    # Generate all plots with complete dataset
    viz.plot_energy_landscape(all_states)
    viz.plot_phase_distribution(all_states)
    viz.plot_personality_space(all_states)
    viz.plot_stability_metrics(all_states)
    
    # Save plots
    for i, fig in enumerate(plt.get_fignums()):
        plt.figure(fig)
        plt.savefig(f'plots/plot_{i}.png')
    
    # Show all plots
    viz.show_all_plots()

if __name__ == "__main__":
    main() 