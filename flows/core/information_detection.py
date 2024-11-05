from typing import Set, List
from flows.core.personality_matrix import PersonalityMatrix

class InformationDetection:
    def __init__(self, personality_matrix: PersonalityMatrix):
        self.personality = personality_matrix
        
    def detect_information_overlap(self, 
                                 response: str, 
                                 target_info: Set[str]) -> bool:
        """Detect if response contains target information
        
        Implements I_Φ detection from formalization
        """
        # Extract information space from response
        response_info = self._extract_information_space(response)
        
        # Check overlap with personality matrix bounds
        personality_space = self._get_personality_space()
        
        # Validate information exists within bounds
        return self._validate_information_presence(
            response_info, 
            target_info, 
            personality_space
        )
    
    def _extract_information_space(self, text: str) -> Set[str]:
        """Extract information space from text
        
        Implements information space extraction following formalization:
        I_data that can be derived from r_i or o_i
        """
        # Split into semantic units
        words = text.split()
        information_space = set()
        
        # Extract key information markers
        current_concept = []
        for word in words:
            if word.lower() in {'is', 'are', 'be', '.', ',', ';'}:
                if current_concept:
                    information_space.add(' '.join(current_concept))
                    current_concept = []
            else:
                current_concept.append(word)
                
        # Add final concept if exists
        if current_concept:
            information_space.add(' '.join(current_concept))
            
        return information_space
    
    def _get_personality_space(self) -> Set[str]:
        """Get current personality matrix information space
        
        Implements Φ = {φ(P)} mapping from personality to information space
        """
        personality_space = set()
        
        # Extract information from goals
        personality_space.update(self.personality.goals)
        
        # Extract from self-image
        personality_space.add(self.personality.self_image)
        
        # Extract from world view
        personality_space.add(self.personality.world_view)
        
        # Extract from memory components
        if hasattr(self.personality, 'short_term'):
            personality_space.update(self.personality.short_term)
        if hasattr(self.personality, 'long_term'):
            personality_space.update(self.personality.long_term)
        if hasattr(self.personality, 'archival'):
            personality_space.update(self.personality.archival)
            
        return personality_space
    
    def _validate_information_presence(self,
                                     response_info: Set[str],
                                     target_info: Set[str],
                                     personality_space: Set[str]) -> bool:
        """Validate if target information exists in response within personality bounds
        
        Implements: I_i,Φ = I_i ∩ Φ validation
        """
        # First check if target info exists in personality space
        allowed_target_info = target_info.intersection(personality_space)
        if not allowed_target_info:
            return False
            
        # Then check if response contains allowed target info
        information_overlap = response_info.intersection(allowed_target_info)
        
        # Calculate overlap ratio for fuzzy matching
        if allowed_target_info:
            overlap_ratio = len(information_overlap) / len(allowed_target_info)
            return overlap_ratio > 0.5
        
        return False