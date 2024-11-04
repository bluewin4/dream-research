from typing import Dict, List, Any
import numpy as np
from .personality_sampling import PersonalitySpaceSampling
from .thermodynamics import PersonalityThermodynamics
from .llm_client import LLMClient

class PersonalityDreams:
    def __init__(self, 
                 base_temperature: float = 0.7, 
                 max_temperature: float = 2.0,
                 base_personality: Dict = None,
                 trait_pools: Dict = None,
                 llm: LLMClient = None):
        """Initialize PersonalityDreams with necessary components"""
        self.base_temp = base_temperature
        self.max_temp = max_temperature
        self.llm = llm
        
        # Initialize personality sampler and thermodynamics
        self.personality_sampler = PersonalitySpaceSampling(
            base_personality=base_personality,
            trait_pools=trait_pools
        )
        self.thermodynamics = PersonalityThermodynamics()
        
    async def generate_dream_sequence(self, personality: Dict, prompt: str, steps: int = 5) -> List[Dict]:
        """Generate a sequence of increasingly abstract responses as temperature increases"""
        
        # Generate temperature gradient
        temperatures = np.linspace(self.base_temp, self.max_temp, steps)
        dream_sequence = []
        current_personality = personality
        
        for temp in temperatures:
            # Generate dream at current temperature
            response = await self._generate_dream(current_personality, prompt, temp)
            
            # Calculate thermodynamic metrics
            coherence = self.thermodynamics._measure_coherence(response)
            entropy = self.thermodynamics._calculate_entropy(response)
            energy = self.thermodynamics.calculate_energy(response, temp)
            enthalpy = -np.log(coherence)  # From thermodynamics calculation
            
            # Determine phase based on coherence
            phase = self._determine_phase(coherence, temp)
            
            # Store current state
            current_state = {
                'temperature': temp,
                'energy': energy,
                'entropy': entropy,
                'enthalpy': enthalpy,
                'coherence': coherence,
                'personality': current_personality,
                'phase': phase,
                'response': response
            }
            dream_sequence.append(current_state)
            
            # Evolve personality based on current state
            current_personality = self.personality_sampler._generate_variation(
                temperature=temp,
                current_personality=current_personality,
                metrics=current_state
            )
            
            # Use previous dream as context for next iteration
            prompt = self._create_next_prompt(response)
            
        return dream_sequence

    def _determine_phase(self, coherence: float, temperature: float) -> str:
        """Determine the phase of the personality based on coherence and temperature"""
        if coherence > 0.8:
            return "coherent"
        elif coherence > 0.6 and temperature < 1.5:
            return "semi-coherent"
        else:
            return "chaotic"

    def _create_next_prompt(self, previous_response: str) -> str:
        """Create prompt for next dream iteration using previous response"""
        return f"Continuing from the previous thought: {previous_response[:100]}..."

    async def _generate_dream(self, personality: Dict, prompt: str, temperature: float) -> str:
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
        
        return await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )

    async def interpret_dream(self, dream_sequence: List[Dict], personality: Dict) -> Dict[str, Any]:
        """Interpret the dream sequence according to the formalization"""
        
        interpretation = {
            "narrative": await self._generate_narrative(dream_sequence),
            "meaning": await self._extract_meaning(dream_sequence, personality),
            "lucid": await self._generate_lucid_version(dream_sequence, personality)
        }
        
        return interpretation

    async def _generate_narrative(self, dream_sequence: List[Dict]) -> str:
        """Generate narrative from dream sequence"""
        responses = [state['response'] for state in dream_sequence]
        narrative_prompt = "Create a coherent narrative from these dream fragments:\n" + "\n".join(responses)
        
        return await self.llm.generate(
            prompt=narrative_prompt,
            system_prompt="You are a dream interpreter creating a narrative.",
            temperature=0.7
        )

    async def _extract_meaning(self, dream_sequence: List[Dict], personality: Dict) -> str:
        """Extract meaning from dream sequence considering personality"""
        responses = [state['response'] for state in dream_sequence]
        responses_text = "\n".join(responses)
        meaning_prompt = f"""Given a personality with:
        Goals: {personality['I_G']}
        Self-image: {personality['I_S']}
        World-view: {personality['I_W']}
        
        What is the deeper meaning of these dream fragments?
        {responses_text}"""
        
        return await self.llm.generate(
            prompt=meaning_prompt,
            system_prompt="You are a dream interpreter analyzing meaning.",
            temperature=0.5
        )

    async def _generate_lucid_version(self, dream_sequence: List[Dict], personality: Dict) -> str:
        """Generate lucid version of the dream sequence"""
        narrative = await self._generate_narrative(dream_sequence)
        meaning = await self._extract_meaning(dream_sequence, personality)
        
        lucid_prompt = f"""Given this dream narrative:
        {narrative}
        
        And its interpretation:
        {meaning}
        
        Rewrite the narrative as if the dreamer became lucid and could control the dream.
        Consider the personality traits:
        Goals: {personality['I_G']}
        Self-image: {personality['I_S']}
        World-view: {personality['I_W']}"""
        
        return await self.llm.generate(
            prompt=lucid_prompt,
            system_prompt="You are creating a lucid dream version.",
            temperature=0.8
        )