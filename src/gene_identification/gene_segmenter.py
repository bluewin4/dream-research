import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

class GeneSegmenter:
    """
    Identifies and segments genes within a prompt using semantic segmentation.
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.label_map = self.model.config.id2label

    def segment_genes(self, prompt: str) -> List[str]:
        """
        Segments the prompt into genes.
        
        Args:
            prompt (str): The input prompt.
        
        Returns:
            List[str]: A list of identified genes.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        genes = []
        current_gene = []
        for token, prediction in zip(tokens, predictions[0]):
            label = self.label_map[prediction.item()]
            if label.startswith("B-"):
                if current_gene:
                    genes.append(" ".join(current_gene))
                    current_gene = []
                current_gene.append(token.replace("##", ""))
            elif label.startswith("I-") and current_gene:
                current_gene.append(token.replace("##", ""))
            else:
                if current_gene:
                    genes.append(" ".join(current_gene))
                    current_gene = []
        
        if current_gene:
            genes.append(" ".join(current_gene))
        
        return genes
