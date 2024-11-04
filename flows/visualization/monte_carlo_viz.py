import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from flows.core.monte_carlo import MCState

class MonteCarloVisualizer:
    def __init__(self):
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # Fallback to default style if seaborn style is not available
            plt.style.use('default')
        
    def plot_energy_landscape(self, states: List[MCState]):
        """Plot energy landscape over temperature"""
        temperatures = [s.temperature for s in states]
        energies = [s.energy for s in states]
        
        plt.figure(1, figsize=(10, 6))
        plt.plot(temperatures, energies, '-o', alpha=0.6)
        plt.xlabel('Temperature')
        plt.ylabel('Energy')
        plt.title('Energy Landscape')
        plt.colorbar(plt.scatter(temperatures, energies, c=energies))
        
    def plot_phase_distribution(self, states: List[MCState]):
        """Plot phase distribution over simulation"""
        phases = [s.phase for s in states]
        unique_phases = list(set(phases))
        
        # Count phases over time
        phase_counts = {phase: [1 if s.phase == phase else 0 for s in states] 
                       for phase in unique_phases}
        
        plt.figure(2, figsize=(12, 6))
        bottom = np.zeros(len(states))
        
        for phase in unique_phases:
            plt.bar(range(len(states)), phase_counts[phase], 
                   bottom=bottom, label=phase, alpha=0.7)
            bottom += phase_counts[phase]
            
        plt.xlabel('Simulation Step')
        plt.ylabel('Phase Distribution')
        plt.title('Phase Distribution Over Time')
        plt.legend()
        
    def plot_personality_space(self, states: List[MCState]):
        """Plot personality space transitions"""
        # Extract personality vectors for each state
        personality_vecs = []
        for state in states:
            vec = []
            for dim in ['I_S', 'I_G', 'I_W']:
                if dim in state.personality:
                    # Simple encoding - could be more sophisticated
                    vec.append(len(str(state.personality[dim])))
            personality_vecs.append(vec)
            
        personality_vecs = np.array(personality_vecs)
        
        # Create 3D plot
        fig = plt.figure(3, figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(personality_vecs[:, 0], 
                           personality_vecs[:, 1],
                           personality_vecs[:, 2],
                           c=[s.temperature for s in states],
                           cmap='viridis')
        
        ax.set_xlabel('Traits (I_S)')
        ax.set_ylabel('Goals (I_G)') 
        ax.set_zlabel('Worldview (I_W)')
        plt.colorbar(scatter, label='Temperature')
        plt.title('Personality Space Transitions')
        
    def plot_stability_metrics(self, states: List[MCState]):
        """Plot stability metrics"""
        temperatures = [s.temperature for s in states]
        
        # Calculate stability metrics
        response_lengths = [len(s.response.split()) for s in states]
        energy_gradient = np.gradient([s.energy for s in states])
        phase_changes = [1 if i > 0 and states[i].phase != states[i-1].phase else 0 
                        for i in range(len(states))]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), num=4)
        
        # Response length
        ax1.plot(temperatures, response_lengths)
        ax1.set_ylabel('Response Length')
        
        # Energy gradient
        ax2.plot(temperatures, energy_gradient)
        ax2.set_ylabel('Energy Gradient')
        
        # Phase changes
        ax3.bar(temperatures, phase_changes, alpha=0.6)
        ax3.set_ylabel('Phase Changes')
        ax3.set_xlabel('Temperature')
        
        plt.tight_layout()
        
    def show_all_plots(self):
        """Display all plots at once"""
        plt.show()