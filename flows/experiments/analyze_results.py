from flows.core.data_storage import DataStorage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import pandas as pd
from flows.core.monte_carlo import MCState
from flows.visualization.phase_separation_viz import PhaseSeparationVisualizer

class ResultAnalyzer:
    def __init__(self):
        self.data_storage = DataStorage()
        
    def analyze_experiment(self, generation_id: str):
        """Analyze experiment results"""
        # Load data
        df, metadata = self.data_storage.load_generation(generation_id)
        
        # Print experiment parameters
        print("Experiment Parameters:")
        for key, value in metadata["parameters"].items():
            print(f"{key}: {value}")
            
        # Convert DataFrame to MCState objects
        states = [
            MCState(
                temperature=row['temperature'],
                energy=row['energy'],
                entropy=row.get('entropy', 0),
                enthalpy=row.get('enthalpy', 0),
                coherence=row.get('coherence', 0),
                phase=row['phase'],
                personality=row.get('personality', {}),
                response=row.get('response', '')
            ) for _, row in df.iterrows()
        ]
        
        # Initialize visualizer
        viz = PhaseSeparationVisualizer()
        
        # Create and show all visualizations
        viz.plot_phase_separation(states)
        viz.plot_free_energy_landscape(states)
        viz.plot_phase_stability_matrix(states)
        
        plt.show()
        
    def compare_experiments(self, generation_ids: List[str]):
        """Compare multiple experiments"""
        dfs = []
        for gen_id in generation_ids:
            df, metadata = self.data_storage.load_generation(gen_id)
            df["experiment"] = metadata["generation_id"]
            dfs.append(df)
            
        combined_df = pd.concat(dfs)
        
        # Create comparison visualizations
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=combined_df, x="experiment", y="energy")
        plt.xticks(rotation=45)
        plt.title("Energy Distribution Across Experiments")
        plt.show() 