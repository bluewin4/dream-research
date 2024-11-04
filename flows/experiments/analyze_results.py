from flows.core.data_storage import DataStorage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import pandas as pd

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
            
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # Energy vs Temperature plot
        plt.subplot(2, 1, 1)
        sns.scatterplot(data=df, x="temperature", y="energy")
        plt.title("Energy vs Temperature")
        
        # Phase distribution
        plt.subplot(2, 1, 2)
        sns.countplot(data=df, x="phase")
        plt.title("Phase Distribution")
        
        plt.tight_layout()
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