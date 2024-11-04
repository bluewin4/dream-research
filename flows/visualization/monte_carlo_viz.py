import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from flows.core.monte_carlo import MCState
from flows.core.personality_embeddings import PersonalityEmbeddingLibrary

class MonteCarloVisualizer:
    def __init__(self):
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        self.embedding_library = PersonalityEmbeddingLibrary()
            
    def plot_thermodynamic_landscape(self, states: List[MCState]):
        """Plot comprehensive thermodynamic landscape"""
        temperatures = [s.temperature for s in states]
        
        # Create subplots for each thermodynamic property
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Thermodynamic Landscape Analysis')
        
        # Energy plot
        ax1.scatter(temperatures, [s.energy for s in states], alpha=0.6, c=temperatures, cmap='viridis')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Temperature')
        
        # Entropy plot
        ax2.scatter(temperatures, [s.entropy for s in states], alpha=0.6, c=temperatures, cmap='viridis')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy vs Temperature')
        
        # Enthalpy plot
        ax3.scatter(temperatures, [s.enthalpy for s in states], alpha=0.6, c=temperatures, cmap='viridis')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Enthalpy')
        ax3.set_title('Enthalpy vs Temperature')
        
        # Coherence plot
        ax4.scatter(temperatures, [s.coherence for s in states], alpha=0.6, c=temperatures, cmap='viridis')
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Coherence')
        ax4.set_title('Coherence vs Temperature')
        
        plt.tight_layout()
        
    def plot_phase_stability(self, states: List[MCState]):
        """Plot phase stability metrics"""
        temperatures = [s.temperature for s in states]
        phases = [s.phase for s in states]
        unique_phases = list(set(phases))
        
        # Create phase transition plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Phase probability distribution
        phase_probs = {}
        temp_bins = np.linspace(min(temperatures), max(temperatures), 20)
        
        for phase in unique_phases:
            phase_probs[phase] = []
            for i in range(len(temp_bins)-1):
                mask = (np.array(temperatures) >= temp_bins[i]) & (np.array(temperatures) < temp_bins[i+1])
                phase_count = sum(1 for p in np.array(phases)[mask] if p == phase)
                prob = phase_count / sum(mask) if sum(mask) > 0 else 0
                phase_probs[phase].append(prob)
        
        # Plot phase probabilities
        for phase in unique_phases:
            ax1.plot(temp_bins[:-1], phase_probs[phase], '-o', label=phase, alpha=0.7)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Phase Probability')
        ax1.set_title('Phase Stability Analysis')
        ax1.legend()
        
        # Plot phase transitions
        transitions = [1 if i > 0 and phases[i] != phases[i-1] else 0 
                      for i in range(len(phases))]
        ax2.scatter(temperatures, transitions, alpha=0.6)
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Phase Transition')
        ax2.set_title('Phase Transitions')
        
        plt.tight_layout()
        
    async def plot_personality_evolution(self, states: List[MCState]):
        """Plot personality vector evolution"""
        # Create two subplots: 3D trajectory and similarity matrix
        fig = plt.figure(figsize=(15, 6))
        
        # 3D Trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Extract personality vectors with semantic encoding
        vectors = []
        for state in states:
            vec = await self._encode_personality_state(state.personality)
            vectors.append(vec)
        
        vectors = np.array(vectors)
        temperatures = np.array([s.temperature for s in states])
        
        # Plot trajectory with lines connecting points
        scatter = ax1.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2],
                           c=temperatures, cmap='viridis',
                           alpha=0.8, s=50)
        
        # Add lines to show evolution trajectory
        ax1.plot(vectors[:, 0], vectors[:, 1], vectors[:, 2],
                 'r-', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Personality Component 1')
        ax1.set_ylabel('Personality Component 2')
        ax1.set_zlabel('Personality Component 3')
        
        # Add colorbar for temperature
        plt.colorbar(scatter, label='Temperature')
        
        # Similarity matrix plot
        ax2 = fig.add_subplot(122)
        
        # Compute similarity matrix
        n_samples = len(vectors)
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i, j] = self.embedding_library.compute_similarity(vectors[i], vectors[j])
        
        # Plot similarity matrix
        im = ax2.imshow(similarity_matrix, cmap='RdYlBu', aspect='auto')
        plt.colorbar(im, label='Cosine Similarity')
        ax2.set_title('Personality State Similarities')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('State Index')
        
        fig.suptitle('Personality Evolution Analysis', y=1.05)
        plt.tight_layout()
        
    async def _encode_personality_state(self, personality: Dict) -> np.ndarray:
        """Encode personality dictionary into a 3D vector using pre-computed embeddings"""
        return await self.embedding_library.compute_personality_vector(personality)
        
    def show_all_plots(self):
        """Display all visualization plots"""
        plt.show()