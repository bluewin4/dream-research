#!/bin/bash

# Create and activate conda environment
conda create -n dream_research python=3.12 -y
conda activate dream_research

# Install requirements
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install package in development mode
pip install -e . 