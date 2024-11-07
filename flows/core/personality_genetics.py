from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Callable
import numpy as np
from .personality_matrix import PersonalityMatrix
from .thermodynamics import PersonalityThermodynamics

@dataclass
class Gene:
    """Fundamental unit of personality information"""
    sequence: str  # The actual information content
    position: int  # Location in personality space
    expression_temp: float  # Temperature at which gene activates
    stability: float  # Resistance to mutation
    phenotype: Optional[str] = None  # Observable trait when expressed
    
@dataclass 
class Chromosome:
    """Collection of related genes forming a personality trait"""
    genes: List[Gene]
    trait_name: str
    activation_threshold: float
    coherence_requirement: float

class PersonalityGenome:
    """Manages the genetic structure of a personality"""
    
    def __init__(self, personality: PersonalityMatrix):
        self.thermo = PersonalityThermodynamics()
        self.chromosomes: Dict[str, Chromosome] = {}
        self.neutral_networks: Dict[str, Set[str]] = {}
        self.stability_cache: Dict[str, float] = {}
        
        # Initialize from personality matrix
        self._initialize_genome(personality)
        
    def _initialize_genome(self, personality: PersonalityMatrix):
        """Convert personality matrix to genetic structure"""
        # Map identity components to chromosomes
        self._map_identity_chromosome(personality.identity)
        # Map memory structures
        self._map_memory_chromosome(personality.memory)
        # Map behavioral structures 
        self._map_structure_chromosome(personality.structure)

    async def mutate(self, temp: float) -> 'PersonalityGenome':
        """Apply temperature-dependent mutations"""
        new_genome = self.copy()
        
        for chromosome in new_genome.chromosomes.values():
            if temp > chromosome.activation_threshold:
                await self._mutate_chromosome(chromosome, temp)
                
        return new_genome

    async def _mutate_chromosome(self, chromosome: Chromosome, temp: float):
        """Apply mutations to a chromosome based on temperature"""
        for gene in chromosome.genes:
            if np.random.random() < self._mutation_probability(gene, temp):
                await self._mutate_gene(gene, temp)

    def _mutation_probability(self, gene: Gene, temp: float) -> float:
        """Calculate probability of mutation based on temperature and stability"""
        base_prob = 1 - gene.stability
        temp_factor = np.exp((temp - gene.expression_temp) / self.thermo.params.T_c)
        return min(base_prob * temp_factor, 1.0)

    async def _mutate_gene(self, gene: Gene, temp: float):
        """Apply temperature-appropriate mutation to gene"""
        if temp < self.thermo.params.T_c:
            # Conservative mutations - maintain meaning
            await self._apply_synonymous_mutation(gene)
        elif temp < self.thermo.phase_boundaries["semi_to_chaotic"]:
            # Semi-stable mutations - allow meaning drift
            await self._apply_missense_mutation(gene)
        else:
            # Chaotic mutations - allow major changes
            await self._apply_nonsense_mutation(gene)

    def measure_robustness(self) -> Dict[str, float]:
        """Measure overall genome stability"""
        robustness_metrics = {
            "global_stability": self._calculate_global_stability(),
            "trait_coherence": self._calculate_trait_coherence(),
            "neutral_network_size": self._calculate_neutral_network_size(),
            "percolation_threshold": self._calculate_percolation_threshold()
        }
        return robustness_metrics

    def _calculate_global_stability(self) -> float:
        """Calculate overall genome stability"""
        stabilities = []
        weights = []
        
        for chromosome in self.chromosomes.values():
            chr_stability = np.mean([g.stability for g in chromosome.genes])
            chr_weight = len(chromosome.genes)
            stabilities.append(chr_stability)
            weights.append(chr_weight)
            
        return np.average(stabilities, weights=weights)

    def _calculate_neutral_network_size(self) -> float:
        """Calculate size of neutral networks"""
        total_size = sum(len(network) for network in self.neutral_networks.values())
        return total_size / len(self.neutral_networks) if self.neutral_networks else 0.0 