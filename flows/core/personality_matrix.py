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
    
    def to_dict(self) -> dict:
        """Convert personality matrix to dictionary format"""
        # Return a dictionary representation of your personality matrix
        # Adjust this based on your actual PersonalityMatrix implementation
        return {
            # Add your personality matrix attributes here
            # For example:
            "traits": self.traits if hasattr(self, 'traits') else {},
            "values": self.values if hasattr(self, 'values') else {},
            # ... other attributes
        }

    def copy(self) -> 'PersonalityMatrix':
        """Create a deep copy of the PersonalityMatrix"""
        new_matrix = PersonalityMatrix()
        new_matrix.memory = Memory(
            short_term=self.memory.short_term.copy(),
            long_term=self.memory.long_term.copy(),
            archival=self.memory.archival.copy()
        )
        new_matrix.structure = Structure(
            input_format=self.structure.input_format,
            tools=self.structure.tools.copy(),
            output_format=self.structure.output_format
        )
        new_matrix.identity = Identity(
            goals=self.identity.goals.copy(),
            methods=self.identity.methods.copy(),
            self_image=self.identity.self_image,
            world_view=self.identity.world_view,
            thoughts=self.identity.thoughts.copy()
        )
        return new_matrix