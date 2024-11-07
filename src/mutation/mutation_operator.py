import random
from typing import List
import re

class MutationOperator:
    """
    Applies mutations to genes within a prompt.
    """
    
    VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate

    def mutate_gene(self, gene: str) -> str:
        """
        Mutates a single gene based on the mutation rate.
        
        Args:
            gene (str): The gene to mutate.
        
        Returns:
            str: The mutated gene.
        """
        if random.random() > self.mutation_rate:
            return gene
        
        mutation_type = random.choice(['synonymous', 'missense', 'nonsense'])
        if mutation_type == 'synonymous':
            return self.synonymous_mutation(gene)
        elif mutation_type == 'missense':
            return self.missense_mutation(gene)
        else:
            return self.nonsense_mutation(gene)

    def synonymous_mutation(self, gene: str) -> str:
        """
        Performs a synonymous mutation (no change in meaning).
        
        Args:
            gene (str): The gene to mutate.
        
        Returns:
            str: The mutated gene.
        """
        # Example: Replace "non-issue" with "no issue"
        synonyms = {
            "non-issue": "no issue",
            "I have a pet dog": "I possess a pet dog",
            "suggest stretching": "recommend stretching"
        }
        return synonyms.get(gene, gene)
    
    def missense_mutation(self, gene: str) -> str:
        """
        Performs a missense mutation (changes the meaning).
        
        Args:
            gene (str): The gene to mutate.
        
        Returns:
            str: The mutated gene.
        """
        # Simple substitution example
        substitutions = {
            "patient": "ptient",
            "cramps": "crams",
            "muscle": "muscly"
        }
        for original, substitute in substitutions.items():
            gene = gene.replace(original, substitute)
        return gene

    def nonsense_mutation(self, gene: str) -> str:
        """
        Performs a nonsense mutation (introduces errors).
        
        Args:
            gene (str): The gene to mutate.
        
        Returns:
            str: The mutated gene.
        """
        # Insert random characters or delete characters
        mutation_choice = random.choice(['insert', 'delete'])
        if mutation_choice == 'insert':
            pos = random.randint(0, len(gene))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return gene[:pos] + char + gene[pos:]
        elif mutation_choice == 'delete' and len(gene) > 1:
            pos = random.randint(0, len(gene)-1)
            return gene[:pos] + gene[pos+1:]
        return gene

    def mutate_prompt(self, prompt: str, genes: List[str]) -> str:
        """
        Applies mutations to the entire prompt.
        
        Args:
            prompt (str): The original prompt.
            genes (List[str]): The list of genes identified in the prompt.
        
        Returns:
            str: The mutated prompt.
        """
        mutated_genes = [self.mutate_gene(gene) for gene in genes]
        for original, mutated in zip(genes, mutated_genes):
            prompt = prompt.replace(original, mutated)
        return prompt

    # Large Scale Mutations

    def invert_gene(self, gene: str) -> str:
        """
        Inverts the gene string.
        
        Args:
            gene (str): The gene to invert.
        
        Returns:
            str: The inverted gene.
        """
        return gene[::-1]

    def duplicate_gene(self, gene: str) -> str:
        """
        Duplicates the gene string.
        
        Args:
            gene (str): The gene to duplicate.
        
        Returns:
            str: The duplicated gene.
        """
        return gene + gene

    def move_gene(self, prompt: str, gene: str, new_position: int) -> str:
        """
        Moves a gene to a new position within the prompt.
        
        Args:
            prompt (str): The original prompt.
            gene (str): The gene to move.
            new_position (int): The position to move the gene to.
        
        Returns:
            str: The mutated prompt with the gene moved.
        """
        prompt = prompt.replace(gene, "")
        words = prompt.split()
        words.insert(new_position, gene)
        return " ".join(words)
