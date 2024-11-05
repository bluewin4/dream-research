from typing import Dict, List, Optional, TypedDict
import random
import numpy as np
from .core.personality_sampling import PersonalityMatrix
from .core.thermodynamics import PersonalityThermodynamics

class PersonalityGenerator:
    def __init__(self):
        # Core trait bags
        self.traits = {
            "analytical": [
                "analyze", "systematic", "logical", "precise", "methodical",
                "rational", "structured", "investigative", "detailed", "objective"
            ],
            "creative": [
                "innovative", "imaginative", "artistic", "expressive", "original",
                "inventive", "experimental", "intuitive", "visionary", "exploratory"
            ],
            "social": [
                "collaborative", "empathetic", "communicative", "supportive", "engaging",
                "interactive", "connecting", "inclusive", "responsive", "understanding"
            ],
            "practical": [
                "efficient", "pragmatic", "reliable", "focused", "consistent",
                "organized", "purposeful", "steady", "grounded", "results-oriented"
            ]
        }
        
        # Environmental perspectives
        self.world_views = {
            "analytical": [
                "system of interconnected principles",
                "framework of logical patterns",
                "structured network of knowledge",
                "complex analytical landscape"
            ],
            "creative": [
                "canvas of endless possibilities",
                "dynamic space of innovation",
                "realm of creative exploration",
                "evolving artistic dimension"
            ],
            "social": [
                "interconnected community",
                "collaborative ecosystem",
                "network of shared experiences",
                "harmonious social fabric"
            ],
            "practical": [
                "organized framework",
                "efficient mechanism",
                "practical foundation",
                "functional environment"
            ]
        }
        
        # Action verbs for goals
        self.goal_verbs = [
            "explore", "develop", "optimize", "create", "analyze",
            "build", "discover", "implement", "investigate", "synthesize"
        ]
        
        # Goal domains
        self.goal_domains = [
            "knowledge", "solutions", "systems", "relationships", "innovations",
            "processes", "understanding", "frameworks", "connections", "patterns"
        ]

    def generate(self) -> PersonalityMatrix:
        """Generate a random personality from trait components"""
        # Randomly select primary and secondary trait categories
        trait_categories = random.sample(list(self.traits.keys()), k=2)
        primary_trait = trait_categories[0]
        secondary_trait = trait_categories[1]
        
        # Generate goals combining verbs, traits, and domains
        goals = []
        for _ in range(4):
            verb = random.choice(self.goal_verbs)
            trait = random.choice(self.traits[random.choice(trait_categories)])
            domain = random.choice(self.goal_domains)
            goals.append(f"{verb} {trait} {domain}")
        
        # Generate self image combining traits
        primary_trait_word = random.choice(self.traits[primary_trait])
        secondary_trait_word = random.choice(self.traits[secondary_trait])
        self_image = f"{primary_trait_word} {secondary_trait_word} system"
        
        # Generate worldview combining perspectives
        primary_view = random.choice(self.world_views[primary_trait])
        secondary_view = random.choice(self.world_views[secondary_trait])
        world_view = f"A {primary_view} with {secondary_view}"
        
        return PersonalityMatrix(
            I_G=goals,
            I_S=self_image,
            I_W=world_view
        )

    def generate_diverse_personalities(self, n_personalities: int) -> List[PersonalityMatrix]:
        """Generate multiple diverse personalities"""
        return [self.generate() for _ in range(n_personalities)]