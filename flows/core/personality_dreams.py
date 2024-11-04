from typing import List, Dict, Any
import numpy as np
from llm_client import LLMClient  # Hypothetical LLM client

class PersonalityDreams:
    def __init__(self, base_temperature: float = 0.7, max_temperature: float = 2.0):
        self.base_temp = base_temperature
        self.max_temp = max_temperature
        self.llm = LLMClient()
    
    def generate_dream_sequence(self, personality: Dict, prompt: str, steps: int = 5) -> List[str]:
        """Generate a sequence of increasingly abstract responses as temperature increases"""
        
        # Generate temperature gradient
        temperatures = np.linspace(self.base_temp, self.max_temp, steps)
        dream_sequence = []
        
        for temp in temperatures:
            # Generate dream at current temperature
            dream = self._generate_dream(personality, prompt, temp)
            dream_sequence.append(dream)
            
            # Use previous dream as context for next iteration
            prompt = self._create_next_prompt(dream)
            
        return dream_sequence
    
    def interpret_dream(self, dream_sequence: List[str], personality: Dict) -> Dict[str, Any]:
        """Interpret the dream sequence according to the formalization"""
        
        # Run interpretation at base temperature for stability
        interpretation = {
            "narrative": self._generate_narrative(dream_sequence),
            "meaning": self._extract_meaning(dream_sequence, personality),
            "lucid": self._generate_lucid_version(dream_sequence, personality)
        }
        
        return interpretation

    def _generate_dream(self, personality: Dict, prompt: str, temperature: float) -> str:
        """Generate single dream response at specified temperature
        
        Following the formalization: φ(P_i, r_j) at temperature T to make o_i,j,dream
        """
        system_prompt = f"""You are a language model with the following personality traits:
        Goals: {personality['I_G']}
        Self-image: {personality['I_S']}
        World-view: {personality['I_W']}
        
        You are in a dream-like state. Your responses should become more abstract 
        and free-associative as the temperature increases.
        
        Current temperature: {temperature}"""
        
        return self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=temperature
        )

    def _create_next_prompt(self, previous_dream: str) -> str:
        """Create prompt for next dream iteration using previous as context
        
        This implements the concept of dream chaining where each dream
        builds on the previous one's information space (I_i,j)
        """
        return f"""Continue this dream sequence, building upon and transforming 
        the following dream elements:

        Previous dream:
        {previous_dream}

        Take these elements and create a new dream sequence that builds upon
        these themes but pushes them further into abstraction. Let the imagery
        and concepts evolve naturally."""

    def _generate_narrative(self, dream_sequence: List[str]) -> str:
        """Create coherent narrative from dream sequence
        
        Implements φ_narrative(P_i, o_i,j,dream) to make o_i,j,narrative
        """
        dreams_combined = "\n---\n".join(dream_sequence)
        
        prompt = f"""Analyze this sequence of dreams and create a coherent narrative 
        that connects them together:

        {dreams_combined}

        Create a story that explains how these dreams connect and flow into each other,
        preserving the key symbols and transformations while making them understandable."""
        
        return self.llm.generate(
            prompt=prompt,
            temperature=self.base_temp  # Use base temp for stability
        )

    def _extract_meaning(self, dream_sequence: List[str], personality: Dict) -> str:
        """Extract meaning according to personality matrix
        
        Implements φ_meaning(P_i, o_i,j,narrative) to make o_i,j,meaning
        """
        narrative = self._generate_narrative(dream_sequence)
        
        prompt = f"""Given a personality with:
        Goals: {personality['I_G']}
        Self-image: {personality['I_S']}
        World-view: {personality['I_W']}
        
        Interpret the meaning of this dream narrative:
        {narrative}
        
        Explain what this dream sequence reveals about the personality's:
        1. Current state
        2. Hidden desires or fears
        3. Potential growth or transformation
        4. Relationship to their goals and self-image"""
        
        return self.llm.generate(
            prompt=prompt,
            temperature=self.base_temp
        )

    def _generate_lucid_version(self, dream_sequence: List[str], personality: Dict) -> str:
        """Generate lucid dream version based on interpretation
        
        Implements φ(P_i, o_i,j,narrative, o_i,j,meaning) to make o_i,j,lucid
        """
        meaning = self._extract_meaning(dream_sequence, personality)
        narrative = self._generate_narrative(dream_sequence)
        
        prompt = f"""Given this dream narrative:
        {narrative}
        
        And its interpretation:
        {meaning}
        
        Rewrite the dream as if the personality became lucid (aware they were dreaming) 
        and could guide the dream toward their goals:
        {personality['I_G']}
        
        Show how they would actively transform the dream elements to better align with their:
        1. Desired self-image: {personality['I_S']}
        2. Ideal world-view: {personality['I_W']}"""
        
        return self.llm.generate(
            prompt=prompt,
            temperature=self.base_temp
        )

    def validate_dream(self, dream: str, personality: Dict) -> bool:
        """Validate if dream output contains expected information (I_Φ)
        
        Implements V_valid(o_i) function from formalization
        """
        required_elements = {
            'personality_traits': any(trait in dream.lower() for trait in personality['I_S'].lower().split()),
            'goal_alignment': any(goal in dream.lower() for goal in personality['I_G'].lower().split()),
            'coherent_narrative': len(dream.split()) > 50  # Simple length check
        }
        
        return all(required_elements.values())