from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib

class DataStorage:
    def __init__(self, base_dir: str = "data"):
        """Initialize data storage system
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "generations"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def save_generation(self, 
                       data: List[Dict[str, Any]], 
                       parameters: Dict[str, Any],
                       experiment_name: str) -> str:
        """Save generation data and parameters
        
        Args:
            data: List of dictionaries containing generation results
            parameters: Dictionary of generation parameters
            experiment_name: Name of the experiment
            
        Returns:
            generation_id: Unique identifier for this generation
        """
        # Create unique identifier for this generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_hash = hashlib.md5(str(parameters).encode()).hexdigest()[:8]
        generation_id = f"{experiment_name}_{timestamp}_{param_hash}"
        
        # Save data as CSV
        df = pd.DataFrame(data)
        data_path = self.data_dir / f"{generation_id}.csv"
        df.to_csv(data_path, index=False)
        
        # Save metadata
        metadata = {
            "generation_id": generation_id,
            "timestamp": timestamp,
            "parameters": parameters,
            "experiment_name": experiment_name,
            "columns": list(df.columns),
            "n_samples": len(df),
            "data_path": str(data_path)
        }
        
        metadata_path = self.metadata_dir / f"{generation_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return generation_id
    
    def load_generation(self, generation_id: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load generation data and metadata
        
        Args:
            generation_id: Unique identifier for the generation
            
        Returns:
            tuple of (DataFrame of generation data, Dictionary of metadata)
        """
        # Load metadata
        metadata_path = self.metadata_dir / f"{generation_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found for generation {generation_id}")
            
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        # Load data
        data_path = Path(metadata["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(f"No data found at {data_path}")
            
        df = pd.DataFrame(pd.read_csv(data_path))
        
        return df, metadata
    
    def list_generations(self, 
                        experiment_name: Optional[str] = None, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available generations with optional filtering
        
        Args:
            experiment_name: Filter by experiment name
            start_date: Filter by start date (YYYYMMDD format)
            end_date: Filter by end date (YYYYMMDD format)
            
        Returns:
            List of metadata dictionaries for matching generations
        """
        generations = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file) as f:
                metadata = json.load(f)
                
            # Apply filters
            if experiment_name and metadata["experiment_name"] != experiment_name:
                continue
                
            if start_date and metadata["timestamp"][:8] < start_date:
                continue
                
            if end_date and metadata["timestamp"][:8] > end_date:
                continue
                
            generations.append(metadata)
            
        return generations 