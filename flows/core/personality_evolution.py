from typing import Callable, List, Dict, Optional
import numpy as np
from .personality_genetics import PersonalityGenome
from .personality_matrix import PersonalityMatrix
from .thermodynamics import PersonalityThermodynamics

class PersonalityEvolution:
    """Manages the evolution of personalities through genetic algorithms"""
    
    def __init__(self, 
                 population_size: int = 10,
                 mutation_rate: float = 0.1,
                 selection_pressure: float = 0.5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.thermo = PersonalityThermodynamics()
        self.generation = 0
        self.population: List[PersonalityGenome] = []
        
    async def initialize_population(self, seed_personality: PersonalityMatrix):
        """Create initial population from seed personality"""
        self.population = []
        
        # Create variations of seed personality
        for _ in range(self.population_size):
            genome = PersonalityGenome(seed_personality)
            temp = np.random.uniform(0.1, 2.0)
            mutated_genome = await genome.mutate(temp)
            self.population.append(mutated_genome)
            
    async def evolve(self, 
                    generations: int,
                    fitness_func: Callable[[PersonalityGenome], float],
                    target_fitness: Optional[float] = None) -> Dict:
        """Evolve population towards target fitness"""
        history = []
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            fitness_scores = [await fitness_func(genome) for genome in self.population]
            max_fitness = max(fitness_scores)
            history.append({
                "generation": gen,
                "max_fitness": max_fitness,
                "avg_fitness": np.mean(fitness_scores),
                "population_diversity": self._calculate_diversity()
            })
            
            # Check termination condition
            if target_fitness and max_fitness >= target_fitness:
                break
                
            # Select parents
            parents = self._select_parents(fitness_scores)
            
            # Create next generation
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                child = await self._crossover(parent1, parent2)
                
                # Apply mutations based on temperature
                temp = self._calculate_evolution_temperature()
                mutated_child = await child.mutate(temp)
                new_population.append(mutated_child)
                
            self.population = new_population
            
        return history

    def _calculate_evolution_temperature(self) -> float:
        """Calculate current evolutionary temperature"""
        base_temp = 0.1 + (self.generation / 100)  # Gradually increase temperature
        diversity = self._calculate_diversity()
        
        # Adjust temperature based on population diversity
        temp_adjustment = np.exp(-diversity)
        
        return min(base_temp * temp_adjustment, 2.0)

    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity of population"""
        if not self.population:
            return 0.0
            
        # Calculate pairwise distances between genomes
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = self._genome_distance(self.population[i], self.population[j])
                distances.append(dist)
                
        return np.mean(distances) if distances else 0.0

    async def _crossover(self, 
                        parent1: PersonalityGenome, 
                        parent2: PersonalityGenome) -> PersonalityGenome:
        """Perform crossover between two parent genomes"""
        # Create new genome from parent1
        child = parent1.copy()
        
        # Crossover chromosomes
        for trait in parent2.chromosomes:
            if trait in child.chromosomes:
                if np.random.random() < 0.5:
                    # 50% chance to inherit chromosome from parent2
                    child.chromosomes[trait] = parent2.chromosomes[trait].copy()
                    
        return child 