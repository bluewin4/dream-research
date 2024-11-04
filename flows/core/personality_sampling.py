import numpy as np
from typing import List, Dict, Union, TypedDict, Optional
import random

class PersonalityMatrix(TypedDict):
    I_G: List[str]  # Goals
    I_S: str        # Self image
    I_W: str        # Worldview

class PersonalitySpaceSampling:
    def __init__(self, base_personality: Dict, trait_pools: Dict):
        """Initialize with base personality and trait pools"""
        self.base = base_personality
        self.trait_pools = trait_pools
        
    def _combine_traits(self, trait1, trait2):
        """Combine two traits by mixing their words"""
        # Convert to string if either trait is a list
        if isinstance(trait1, list):
            trait1 = ' '.join(trait1)
        if isinstance(trait2, list):
            trait2 = ' '.join(trait2)
        
        words1 = trait1.split()
        words2 = trait2.split()
        
        # Take key words from both traits
        key_words = [w for w in words1 + words2 
                    if w not in {'and', 'the', 'to', 'a', 'an'}]
                    
        # Combine randomly but maintain grammar
        if len(key_words) > 4:
            key_words = random.sample(key_words, 4)
            
        return ' '.join(key_words)
        
    def _enhance_trait(self, trait: str) -> str:
        """Enhance a trait based on success"""
        enhancers = [
            "effectively",
            "adaptively", 
            "creatively",
            "systematically"
        ]
        return f"{trait} {random.choice(enhancers)}"

    def _generate_variation(self, temperature: float, 
                          current_personality: Optional[PersonalityMatrix] = None,
                          metrics: Optional[Dict] = None) -> PersonalityMatrix:
        """Generate personality variation at given temperature"""
        # If no current personality or metrics, use base temperature sampling
        if current_personality is None or metrics is None:
            return self._base_temperature_sampling(temperature)
            
        # Calculate mutation rate based on metrics
        mutation_rate = temperature * metrics.get('entropy', 0) * (1 - metrics.get('coherence', 0))
        
        # Evolve goals
        new_goals = []
        for goal in current_personality["I_G"]:
            if random.random() < mutation_rate:
                # Mutate goal by combining with another
                other_goal = random.choice(self.trait_pools["goals"])
                new_goal = self._combine_traits(goal, other_goal)
                new_goals.append(new_goal)
            else:
                new_goals.append(goal)
                
        # Evolve self image based on coherence
        coherence = metrics.get('coherence', 0)
        if coherence > 0.8:
            self_image = self._enhance_trait(current_personality["I_S"])
        elif coherence < 0.5:
            self_image = random.choice(self.trait_pools["self_image"])
        else:
            self_image = current_personality["I_S"]
            
        # Evolve worldview based on entropy
        entropy = metrics.get('entropy', 0)
        if entropy > 4.0:
            # High entropy = more dynamic worldview
            worldview = self._combine_traits(
                current_personality["I_W"],
                random.choice(self.trait_pools["worldview"])
            )
        else:
            worldview = current_personality["I_W"]
            
        return PersonalityMatrix(
            I_G=new_goals,
            I_S=self_image,
            I_W=worldview
        )

    def _base_temperature_sampling(self, temperature: float) -> PersonalityMatrix:
        """Original temperature-based sampling logic"""
        if temperature < 0.5:
            # Low temperature: subtle variations
            goals = random.choice(self.trait_pools["goals"])
            self_image = self.base["I_S"]
            worldview = self.base["I_W"]
        elif temperature < 1.0:
            # Medium temperature: moderate variations
            goals = random.choice(self.trait_pools["goals"])
            self_image = random.choice(self.trait_pools["self_image"])
            worldview = self.base["I_W"]
        elif temperature < 1.5:
            # High temperature: significant variations
            goals = random.choice(self.trait_pools["goals"])
            self_image = random.choice(self.trait_pools["self_image"])
            worldview = random.choice(self.trait_pools["worldview"])
        else:
            # Very high temperature: completely random
            goals = random.choice(self.trait_pools["goals"])
            self_image = random.choice(self.trait_pools["self_image"])
            worldview = random.choice(self.trait_pools["worldview"])
            # Add some chaos at very high temperatures
            if random.random() < 0.3:
                goals = [g[::-1] for g in goals]  # reverse strings for chaos

        return PersonalityMatrix(
            I_G=goals,
            I_S=self_image,
            I_W=worldview
        )