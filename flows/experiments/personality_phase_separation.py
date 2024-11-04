from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from ..core.monte_carlo import MonteCarloAnalyzer
from ..core.thermodynamics import PersonalityThermodynamics
from pathlib import Path
import json
from datetime import datetime

class PersonalityPhaseExperiment:
    def __init__(self):
        self.thermodynamics = PersonalityThermodynamics()
        self.mc_analyzer = MonteCarloAnalyzer(self.thermodynamics)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_phase_separation_experiment(self,
                                      personalities: List[Dict],
                                      prompts: List[str],
                                      n_steps: int = 1000,
                                      temp_range: tuple = (0.1, 2.0)) -> Dict:
        """Run phase separation experiment across multiple personalities
        
        Tests if Pr(φ(P,r_ideal) ∈ Φ_i) > Σ Pr(φ(P,r_ideal) ∈ Φ_j)
        """
        results = {
            'phase_probabilities': [],
            'free_energies': [],
            'stability_metrics': [],
            'phase_transitions': []
        }
        
        # Generate temperature schedule
        temps = np.linspace(temp_range[0], temp_range[1], n_steps)
        
        for personality in personalities:
            personality_results = []
            
            for prompt in prompts:
                # Run Monte Carlo simulation
                states = self.mc_analyzer.run_simulation(
                    initial_personality=personality,
                    prompt=prompt,
                    n_steps=n_steps,
                    temperature_schedule=temps
                )
                
                # Analyze phase separation
                phase_probs = self.mc_analyzer.analyze_phase_separation(
                    states=states,
                    personality_spaces=personalities
                )
                
                # Calculate free energy
                free_energy = self.mc_analyzer.calculate_free_energy(
                    states=states,
                    temperature=np.mean(temps)
                )
                
                personality_results.append({
                    'prompt': prompt,
                    'phase_probabilities': phase_probs,
                    'free_energy': free_energy,
                    'states': states
                })
                
            results['phase_probabilities'].append(personality_results)
            
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize phase separation experiment results"""
        # Plot phase probabilities
        self._plot_phase_probabilities(results['phase_probabilities'])
        
        # Plot free energy landscape
        self._plot_free_energy(results['phase_probabilities'])
        
        # Plot phase transitions
        self._plot_phase_transitions(results['phase_probabilities'])
        
    def _plot_phase_probabilities(self, phase_results: List):
        """Plot phase separation probabilities"""
        plt.figure(figsize=(10, 6))
        
        for i, personality_results in enumerate(phase_results):
            probs = [r['phase_probabilities'] for r in personality_results]
            plt.plot(probs, label=f'Personality {i+1}')
            
        plt.xlabel('Temperature')
        plt.ylabel('Phase Probability')
        plt.title('Personality Phase Separation')
        plt.legend()
        plt.show()
        
    def _plot_free_energy(self, phase_results: List):
        """Plot free energy landscape"""
        plt.figure(figsize=(10, 6))
        
        for i, personality_results in enumerate(phase_results):
            energies = [r['free_energy'] for r in personality_results]
            plt.plot(energies, label=f'Personality {i+1}')
            
        plt.xlabel('Temperature')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Landscape')
        plt.legend()
        plt.show()
        
    def _plot_phase_transitions(self, phase_results: List):
        """Plot phase transitions"""
        plt.figure(figsize=(10, 6))
        
        for i, personality_results in enumerate(phase_results):
            states = [s for r in personality_results for s in r['states']]
            phases = [s.phase for s in states]
            
            # Count phase transitions
            transitions = np.diff([hash(p) for p in phases]) != 0
            plt.plot(transitions, label=f'Personality {i+1}')
            
        plt.xlabel('Step')
        plt.ylabel('Phase Transition')
        plt.title('Phase Transitions')
        plt.legend()
        plt.show()
        
    def save_results(self, results: Dict, filename: Optional[str] = None) -> Path:
        """Save experiment results to JSON file
        
        Args:
            results: Dictionary containing experiment results
            filename: Optional custom filename, defaults to timestamp
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase_experiment_{timestamp}.json"
            
        output_path = self.results_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._prepare_for_serialization(results)
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {output_path}")
        return output_path
    
    def load_results(self, filename: str) -> Dict:
        """Load experiment results from JSON file
        
        Args:
            filename: Name of results file to load
            
        Returns:
            Dictionary containing experiment results
        """
        input_path = self.results_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"No results file found at {input_path}")
            
        with open(input_path) as f:
            results = json.load(f)
            
        return results
    
    def list_results(self) -> List[str]:
        """List all saved result files
        
        Returns:
            List of result filenames
        """
        return [f.name for f in self.results_dir.glob("phase_experiment_*.json")]
    
    def _prepare_for_serialization(self, results: Dict) -> Dict:
        """Prepare results dictionary for JSON serialization
        
        Converts numpy arrays and other non-serializable types to basic Python types
        """
        processed = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            elif isinstance(value, list):
                processed[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item 
                    for item in value
                ]
            elif isinstance(value, dict):
                processed[key] = self._prepare_for_serialization(value)
            else:
                processed[key] = value
        return processed 