from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from flows.core.monte_carlo import MCState

class PhaseSeparationVisualizer:
    def __init__(self):
        self.style_config = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        }
        plt.style.use('classic')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value

    def plot_phase_separation(self, states: List[MCState]):
        """Plot phase separation probabilities and transitions"""
        temperatures = [s.temperature for s in states]
        phases = [s.phase for s in states]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Phase separation probability
        unique_phases = list(set(phases)) if phases else ['coherent']
        if len(temperatures) < 2:
            print("Not enough data points for visualization")
            return fig
        
        temp_bins = np.linspace(min(temperatures), max(temperatures), 30)
        bin_centers = (temp_bins[:-1] + temp_bins[1:]) / 2  # Use bin centers for plotting
        
        plotted_something = False  # Flag to track if we plotted anything
        
        for phase in unique_phases:
            probs = np.zeros(len(temp_bins) - 1)
            errors = np.zeros(len(temp_bins) - 1)
            
            for i in range(len(temp_bins)-1):
                mask = (np.array(temperatures) >= temp_bins[i]) & \
                      (np.array(temperatures) < temp_bins[i+1])
                if sum(mask) == 0:
                    continue
                
                phase_count = sum(1 for p in np.array(phases)[mask] if p == phase)
                total = sum(mask)
                probs[i] = phase_count / total if total > 0 else 0
                errors[i] = np.sqrt(probs[i] * (1-probs[i]) / total) if total > 0 and probs[i] > 0 else 0
            
            if np.any(probs > 0):
                ax1.plot(bin_centers, probs, '-', label=f'Phase: {phase}', alpha=0.7)
                ax1.fill_between(bin_centers, 
                               probs - errors,
                               probs + errors,
                               alpha=0.2)
                plotted_something = True
        
        if not plotted_something:
            ax1.text(0.5, 0.5, f'All states are in {unique_phases[0]} phase', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax1.transAxes)
        else:
            ax1.legend()
        
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Phase Separation Probability')
        ax1.set_title('Phase Separation Analysis')
        
        # Phase transition density
        transitions = [1 if i > 0 and phases[i] != phases[i-1] else 0 
                      for i in range(len(phases))]
        
        if sum(transitions) > 1:  # Only compute KDE if we have transitions
            kde = gaussian_kde(np.array(temperatures)[np.array(transitions) == 1])
            x_range = np.linspace(min(temperatures), max(temperatures), 100)
            density = kde(x_range)
            density = density / np.trapz(density, x_range)  # Normalize to 1
            
            ax2.plot(x_range, density)
            ax2.fill_between(x_range, density, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No phase transitions detected', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
        
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Transition Density')
        ax2.set_title('Phase Transition Density')
        
        plt.tight_layout()
        return fig

    def plot_free_energy_landscape(self, states: List[MCState]):
        temperatures = [s.temperature for s in states]
        energies = [s.energy for s in states]
        phases = [s.phase for s in states]
        entropies = [s.entropy for s in states]  # Add new metrics
        enthalpies = [s.enthalpy for s in states]
        coherences = [s.coherence for s in states]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Energy vs Temperature plot
        self._scatter_plot(ax1, temperatures, energies, phases, 
                          'Temperature', 'Energy', 'Energy Landscape')
        
        # Entropy vs Temperature plot
        self._scatter_plot(ax2, temperatures, entropies, phases,
                          'Temperature', 'Entropy', 'Entropy Analysis')
        
        # Enthalpy vs Temperature plot
        self._scatter_plot(ax3, temperatures, enthalpies, phases,
                          'Temperature', 'Enthalpy', 'Enthalpy Analysis')
        
        # Coherence vs Temperature plot
        self._scatter_plot(ax4, temperatures, coherences, phases,
                          'Temperature', 'Coherence', 'Coherence Analysis')
        
        plt.tight_layout()
        return fig

    def _scatter_plot(self, ax, x, y, phases, xlabel, ylabel, title):
        """Helper method for creating scatter plots with phase coloring"""
        phase_styles = {
            'coherent': {'color': 'blue', 'marker': 'o', 'label': 'Coherent'},
            'chaotic': {'color': 'red', 'marker': '^', 'label': 'Chaotic'},
            'semi-coherent': {'color': 'green', 'marker': 's', 'label': 'Semi-coherent'}
        }
        
        # Plot points for each phase
        for phase, style in phase_styles.items():
            mask = np.array(phases) == phase
            if sum(mask) > 0:  # Only plot if we have points for this phase
                ax.scatter(np.array(x)[mask], 
                          np.array(y)[mask],
                          c=style['color'],
                          marker=style['marker'],
                          label=style['label'],
                          alpha=0.6)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    def plot_phase_stability_matrix(self, states: List[MCState]):
        """Plot phase stability matrix showing transition probabilities"""
        phases = [s.phase for s in states]
        # Reorder phases to put semi-coherent in the middle
        phase_order = ['coherent', 'semi-coherent', 'chaotic']
        n_phases = len(phase_order)
        
        # Create transition matrix with ordered phases
        transition_matrix = np.zeros((n_phases, n_phases))
        phase_to_idx = {phase: i for i, phase in enumerate(phase_order)}
        
        for i in range(len(phases)-1):
            if phases[i] in phase_to_idx and phases[i+1] in phase_to_idx:
                current_idx = phase_to_idx[phases[i]]
                next_idx = phase_to_idx[phases[i+1]]
                transition_matrix[current_idx, next_idx] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    where=row_sums!=0)
        
        # Plot with consistent ordering
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(transition_matrix, cmap='viridis')
        
        # Set both axes with the same order
        ax.set_xticks(range(n_phases))
        ax.set_yticks(range(n_phases))
        ax.set_xticklabels(phase_order)
        ax.set_yticklabels(phase_order)
        
        # Add labels
        ax.set_title('Phase Stability Matrix')
        ax.set_xlabel('To Phase')    # x-axis = columns = destination phase
        ax.set_ylabel('From Phase')  # y-axis = rows = starting phase
        
        # Add colorbar
        plt.colorbar(im, label='Transition Probability')
        
        # Add statistical uncertainty
        transition_counts = transition_matrix * row_sums
        uncertainty = np.zeros_like(transition_matrix)
        for i in range(n_phases):
            for j in range(n_phases):
                if row_sums[i] > 0:
                    p = transition_matrix[i,j]
                    n = row_sums[i][0]
                    uncertainty[i,j] = np.sqrt(p * (1-p) / n)
        
        # Add annotations with uncertainties
        for i in range(n_phases):
            for j in range(n_phases):
                if transition_matrix[i,j] > 0:
                    text = f'{transition_matrix[i,j]:.2f}\n±{uncertainty[i,j]:.2f}'
                    ax.text(j, i, text, ha='center', va='center')
        
        plt.tight_layout()
        return fig 

def generate_visualizations(states_file: str, output_dir: str):
    """Generate and save all visualizations for a given states file"""
    import json
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load states from JSON
    with open(states_file, 'r') as f:
        states_data = json.load(f)
    
    # Convert JSON data to MCState-like objects
    class MCState:
        def __init__(self, data):
            self.temperature = data['temperature']
            self.energy = data['energy']
            self.entropy = data['entropy']
            self.enthalpy = data['enthalpy']
            self.coherence = data['coherence']
            self.phase = data['phase']
    
    states = [MCState(state) for state in states_data]
    
    # Create visualizer
    viz = PhaseSeparationVisualizer()
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate and save each visualization
    try:
        # Phase separation plot
        fig_phase = viz.plot_phase_separation(states)
        fig_phase.savefig(os.path.join(output_dir, f'phase_separation_{timestamp}.png'))
        plt.close(fig_phase)
        
        # Free energy landscape
        fig_energy = viz.plot_free_energy_landscape(states)
        fig_energy.savefig(os.path.join(output_dir, f'energy_landscape_{timestamp}.png'))
        plt.close(fig_energy)
        
        # Phase stability matrix
        fig_stability = viz.plot_phase_stability_matrix(states)
        fig_stability.savefig(os.path.join(output_dir, f'phase_stability_{timestamp}.png'))
        plt.close(fig_stability)
        
        print(f"Visualizations generated successfully in {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

if __name__ == "__main__":
    # Example usage
    states_file = "data/generations/phase_exp_1730757604.json"
    output_dir = "outputs/visualizations"
    generate_visualizations(states_file, output_dir)