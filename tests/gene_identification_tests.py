import unittest
from src.gene_identification.gene_segmenter import GeneSegmenter

class TestGeneSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = GeneSegmenter()

    def test_segment_genes_basic(self):
        prompt = "patient is 27 years old and has muscle cramps."
        genes = self.segmenter.segment_genes(prompt)
        self.assertIn("patient", genes)
        self.assertIn("27 years old", genes)
        self.assertIn("muscle cramps", genes)

    def test_segment_genes_no_genes(self):
        prompt = "This prompt has no identifiable genes."
        genes = self.segmenter.segment_genes(prompt)
        self.assertEqual(len(genes), 0)

    def test_segment_genes_complex(self):
        prompt = "The patient is 27, has muscle cramps, and exhibits high blood pressure."
        genes = self.segmenter.segment_genes(prompt)
        self.assertIn("patient", genes)
        self.assertIn("27", genes)
        self.assertIn("muscle cramps", genes)
        self.assertIn("high blood pressure", genes)

if __name__ == '__main__':
    unittest.main()
