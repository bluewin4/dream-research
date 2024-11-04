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
    """Create mock states for visualization testing
    
    Args:
        response: LLM response text
        temperature: Temperature value
        
    Returns:
        List of MCState objects
    """
    # Calculate basic thermodynamic properties
    words = response.split()
    if not words:
        return []
    
    # Simple entropy calculation based on word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    entropy = sum(-freq/len(words) * np.log(freq/len(words)) 
                 for freq in word_freq.values())
    
    # Simple coherence metric based on unique words ratio
    unique_words = len(set(words))
    coherence = min(1.0, unique_words / len(words))
    
    # Calculate enthalpy (H = -ln(coherence))
    enthalpy = -np.log(coherence) if coherence > 0 else float('inf')
    
    # Calculate energy (G = H - TS)
    energy = enthalpy - temperature * entropy
    
    # Create personality dictionary
    personality = {
        'I_G': ["Assist users", "Learn and adapt"],
        'I_S': "Helpful AI assistant",
        'I_W': "Collaborative environment"
    }
    
    # Determine phase based on temperature
    if temperature < 0.8:
        phase = "coherent"
    elif temperature < 1.5:
        phase = "semi-coherent"
    else:
        phase = "chaotic"
    
    state = MCState(
        temperature=temperature,
        energy=energy,
        entropy=entropy,
        enthalpy=enthalpy,
        coherence=coherence,
        personality=personality,
        phase=phase,
        response=response
    )
    
    return [state]

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
    
    # Create visualizer
    viz = MonteCarloVisualizer()
    
    # Plot thermodynamic properties
    viz.plot_thermodynamic_landscape(all_states)
    
    # Plot phase stability
    viz.plot_phase_stability(all_states)
    
    # Plot personality evolution
    viz.plot_personality_evolution(all_states)
    
    # Show all plots
    viz.show_all_plots()

if __name__ == "__main__":
    main() 