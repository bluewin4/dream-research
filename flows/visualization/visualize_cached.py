import json
import asyncio
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from flows.visualization.phase_separation_viz import PhaseSeparationVisualizer
from flows.visualization.monte_carlo_viz import MonteCarloVisualizer
from flows.core.monte_carlo import MCState

async def main():
    # Read generations
    generations_dir = Path('data/generations')
    all_states = []
    
    for file_path in generations_dir.glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Extract metadata and states
                    metadata = data.get('metadata', {})
                    states = data.get('states', [])
                    
                    # Get experiment parameters
                    experiment_id = metadata.get('experiment_id')
                    timestamp = metadata.get('timestamp')
                    parameters = metadata.get('parameters', {})
                    model_info = metadata.get('model', {})
                    
                    for state in states:
                        # Create MCState object from each state entry
                        mc_state = MCState(
                            temperature=float(state['temperature']),
                            energy=float(state['energy']),
                            entropy=float(state['entropy']),
                            enthalpy=float(state['enthalpy']),
                            coherence=float(state['coherence']),
                            personality=state['personality'],
                            phase=state['phase'],
                            response=state['response']
                        )
                        all_states.append(mc_state)
                        print(f"Processed state with temperature: {mc_state.temperature}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON file: {file_path}")
                except KeyError as e:
                    print(f"Warning: Missing required field {e} in file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue

    if not all_states:
        print("No valid states were loaded. Please check your data files.")
        return
        
    print(f"\nAnalysis complete. Generated {len(all_states)} total states")
    # Experiment metadata can be used for plot titles/labels
    print(f"Experiment ID: {experiment_id}")
    print(f"Timestamp: {timestamp}")
    
    # Initialize visualizers
    mc_viz = MonteCarloVisualizer()
    phase_viz = PhaseSeparationVisualizer()
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    # Monte Carlo visualizations
    mc_viz.plot_thermodynamic_landscape(all_states)
    mc_viz.plot_phase_stability(all_states)
    await mc_viz.plot_personality_evolution(all_states)
    
    # Phase separation visualizations
    phase_viz.plot_phase_separation(all_states)
    phase_viz.plot_free_energy_landscape(all_states)
    phase_viz.plot_phase_stability_matrix(all_states)
    
    # Save all plots
    save_plots("phase_analysis")
    
    print("\nVisualization complete. Plots saved to 'plots' directory.")
    
    # Show all plots
    plt.show()

def save_plots(experiment_name: str):
    """Save all plots to files with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Save each figure with a descriptive name
    for i, fig in enumerate(plt.get_fignums()):
        figure = plt.figure(fig)
        plot_type = {
            0: "thermodynamic_landscape",
            1: "phase_stability",
            2: "personality_evolution",
            3: "phase_separation",
            4: "free_energy",
            5: "stability_matrix"
        }.get(i, f"plot_{i}")
        
        filename = f"{experiment_name}_{plot_type}_{timestamp}.png"
        filepath = plots_dir / filename
        figure.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {plot_type} plot to {filepath}")

if __name__ == "__main__":
    asyncio.run(main())