from typing import Dict, List, Optional, TypedDict
import random
import numpy as np
from .core.personality_sampling import PersonalityMatrix
from .core.thermodynamics import PersonalityThermodynamics

class PersonalityGenerator:
    def __init__(self, thermodynamics: Optional[PersonalityThermodynamics] = None):
        # Core trait categories with weighted attributes
        self.traits = {
            "analytical": {
                "attributes": ["Detail-oriented", "Systematic", "Logical", "Methodical", "Precise"],
                "weight": 0.8
            },
            "creative": {
                "attributes": ["Innovative", "Imaginative", "Original", "Artistic", "Experimental"],
                "weight": 0.6
            },
            "social": {
                "attributes": ["Collaborative", "Empathetic", "Supportive", "Communicative", "Interactive"],
                "weight": 0.7
            },
            "practical": {
                "attributes": ["Efficient", "Organized", "Reliable", "Solution-focused", "Pragmatic"],
                "weight": 0.75
            }
        }
        
        # Domain specializations with relevance scores
        self.domains = {
            "Technical problem-solving": 0.9,
            "Creative ideation": 0.7,
            "Data analysis": 0.85,
            "Process optimization": 0.8,
            "Knowledge synthesis": 0.75,
            "Strategic planning": 0.7,
            "Learning systems": 0.8,
            "Research and development": 0.75
        }
        
        # Work styles with compatibility matrices
        self.work_styles = {
            "Structured": {
                "attributes": ["Methodical", "Organized", "Precise"],
                "compatibility": {
                    "analytical": 0.9,
                    "creative": 0.4,
                    "social": 0.6,
                    "practical": 0.8
                }
            },
            "Adaptive": {
                "attributes": ["Flexible", "Responsive", "Dynamic"],
                "compatibility": {
                    "analytical": 0.6,
                    "creative": 0.8,
                    "social": 0.7,
                    "practical": 0.6
                }
            },
            "Collaborative": {
                "attributes": ["Interactive", "Team-oriented", "Supportive"],
                "compatibility": {
                    "analytical": 0.5,
                    "creative": 0.7,
                    "social": 0.9,
                    "practical": 0.6
                }
            }
        }

        self.thermodynamics = thermodynamics or PersonalityThermodynamics()

        # Define base personality (replacing default_personality.json)
        self.base_personality = PersonalityMatrix(
            I_G=[
                "Analyze and solve problems",
                "Learn from interactions", 
                "Generate creative solutions",
                "Optimize processes"
            ],
            I_S="Adaptive problem-solving system",
            I_W="Dynamic knowledge-driven environment focused on growth and innovation"
        )

    def generate(self, temperature: float = 0.7, bias: Optional[Dict[str, float]] = None) -> PersonalityMatrix:
        """Generate personality with temperature-based sampling and optional bias"""
        # Get base personality components
        scaled_weights = self._scale_weights(temperature)
        if bias:
            scaled_weights = self._apply_bias(scaled_weights, bias)
        
        selected_traits = self._sample_traits(scaled_weights)
        selected_domains = self._select_domains(selected_traits, temperature)
        selected_style = self._select_work_style(selected_traits, temperature)
        
        # Create personality matrix
        personality = PersonalityMatrix(
            I_G=selected_traits,
            I_S=selected_domains[0],  # Use primary domain as self-image
            I_W=f"{selected_style} environment focused on {selected_domains[1]}"
        )
        
        # Calculate initial thermodynamic properties
        if self.thermodynamics:
            # Use empty response for initial state
            empty_response = ""
            coherence = self.thermodynamics._measure_coherence(empty_response)
            entropy = self.thermodynamics._calculate_entropy(empty_response)
            energy = self.thermodynamics.calculate_energy(empty_response, temperature)
            
            # Store metrics in personality object
            personality["metrics"] = {
                "coherence": coherence,
                "entropy": entropy,
                "energy": energy,
                "temperature": temperature
            }
        
        return personality

    def _scale_weights(self, temperature: float) -> Dict[str, float]:
        """Scale trait weights based on temperature"""
        scaled = {}
        for category, info in self.traits.items():
            # Higher temperature = more uniform weights
            if temperature > 1.0:
                scaled[category] = 1.0 - (temperature - 1.0) * 0.5  # Gradually flatten
            else:
                scaled[category] = info["weight"] * (1.0 / temperature)
        return scaled

    def _apply_bias(self, weights: Dict[str, float], bias: Dict[str, float]) -> Dict[str, float]:
        """Apply optional bias to trait weights"""
        biased = weights.copy()
        for category, bias_value in bias.items():
            if category in biased:
                biased[category] *= bias_value
        return biased

    def _sample_traits(self, weights: Dict[str, float]) -> List[str]:
        """Sample traits based on weighted categories"""
        # Select categories proportional to weights
        categories = list(weights.keys())
        probs = np.array(list(weights.values()))
        probs = probs / probs.sum()  # Normalize
        
        selected_cats = np.random.choice(
            categories, 
            size=min(3, len(categories)),
            p=probs,
            replace=False
        )
        
        # Sample traits from selected categories
        selected_traits = []
        for category in selected_cats:
            traits = self.traits[category]["attributes"]
            selected_traits.extend(random.sample(traits, k=2))
            
        return selected_traits

    def _select_domains(self, traits: List[str], temperature: float) -> List[str]:
        """Select domains based on traits and temperature"""
        # Calculate domain affinities based on traits
        affinities = {domain: score for domain, score in self.domains.items()}
        
        # Apply temperature effects
        if temperature > 1.0:
            # Higher temperature = more random selection
            domains = random.sample(list(affinities.keys()), k=2)
        else:
            # Lower temperature = select highest affinity
            sorted_domains = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
            domains = [d[0] for d in sorted_domains[:2]]
            
        return domains

    def _select_work_style(self, traits: List[str], temperature: float) -> str:
        """Select work style based on trait compatibility"""
        if temperature > 1.5:
            # Very high temperature = random style
            return random.choice(list(self.work_styles.keys()))
            
        # Calculate compatibility scores
        scores = {}
        for style, info in self.work_styles.items():
            score = sum(info["compatibility"].get(trait, 0.5) for trait in traits)
            scores[style] = score
            
        # Select based on scores and temperature
        if temperature < 0.5:
            # Low temperature = highest compatibility
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Medium temperature = weighted random
            styles = list(scores.keys())
            weights = list(scores.values())
            return random.choices(styles, weights=weights, k=1)[0] 

    def generate_diverse_personalities(self, n_personalities: int, temperature: float = 0.7) -> List[PersonalityMatrix]:
        """Generate diverse personalities using different trait biases"""
        personalities = []
        
        # Define bias profiles to encourage diversity
        bias_profiles = [
            {   # Analytical focus
                "analytical": 0.9,
                "creative": 0.4,
                "social": 0.3,
                "practical": 0.6
            },
            {   # Creative focus
                "analytical": 0.4,
                "creative": 0.9,
                "social": 0.6,
                "practical": 0.3
            },
            {   # Social focus
                "analytical": 0.3,
                "creative": 0.6,
                "social": 0.9,
                "practical": 0.4
            },
            {   # Practical focus
                "analytical": 0.6,
                "creative": 0.3,
                "social": 0.4,
                "practical": 0.9
            },
            {   # Balanced
                "analytical": 0.7,
                "creative": 0.7,
                "social": 0.7,
                "practical": 0.7
            }
        ]
        
        # Generate personalities with different biases
        for i in range(n_personalities):
            bias = bias_profiles[i % len(bias_profiles)]
            personality = self.generate(
                temperature=temperature,
                bias=bias
            )
            personalities.append(personality)
            
        return personalities