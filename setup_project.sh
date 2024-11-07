#!/bin/bash

# Create root directory
mkdir -p project-root
cd project-root

# Create main directories
mkdir -p data/{raw,processed,benchmarks}
mkdir -p src/{gene_identification,mutation,evaluation,models,utils}
mkdir -p tests
mkdir -p notebooks

# Create Python files in src subdirectories
touch src/gene_identification/{__init__,gene_segmenter}.py
touch src/mutation/{__init__,mutation_operator}.py
touch src/evaluation/{__init__,robustness_evaluator}.py
touch src/models/{__init__,lm_handler}.py
touch src/utils/{__init__,helpers}.py

# Create test files
touch tests/{gene_identification,mutation,evaluation}_tests.py

# Create notebook
touch notebooks/exploratory_analysis.ipynb

# Create root level files
touch requirements.txt README.md setup.py

echo "Project structure created successfully!"

# Optional: Print the directory structure
echo -e "\nCreated directory structure:"
tree