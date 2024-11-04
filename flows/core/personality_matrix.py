from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Memory:
    short_term: List[str]  # M_S - verbose context
    long_term: List[str]   # M_L - conversation summaries
    archival: List[str]    # M_A - archivist information

@dataclass
class Structure:
    input_format: str      # S_I - information parsing labels
    tools: List[str]       # S_T - available tools
    output_format: str     # S_O - output structure/guardrails

@dataclass
class Identity:
    goals: List[str]       # I_G - model goals
    methods: List[str]     # I_M - planned approaches
    self_image: str        # I_S - self perception
    world_view: str        # I_W - environment perception
    thoughts: List[str]    # I_T - general thoughts

class PersonalityMatrix:
    def __init__(self):
        self.memory: Memory = Memory([], [], [])
        self.structure: Structure = Structure("", [], "")
        self.identity: Identity = Identity([], [], "", "", [])
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityMatrix':
        """Create PersonalityMatrix from dictionary format"""
        matrix = cls()
        if 'I_G' in data:
            matrix.identity.goals = data['I_G'] if isinstance(data['I_G'], list) else [data['I_G']]
        if 'I_S' in data:
            matrix.identity.self_image = data['I_S']
        if 'I_W' in data:
            matrix.identity.world_view = data['I_W']
        return matrix
    
    def to_dict(self) -> Dict:
        """Convert PersonalityMatrix to dictionary format"""
        return {
            'I_G': self.identity.goals,
            'I_S': self.identity.self_image,
            'I_W': self.identity.world_view
        }