from setuptools import setup, find_packages

setup(
    name="dream_research",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'numpy',
        'matplotlib',
        'openai',
        'spacy',
        'tenacity'
    ],
    extras_require={
        'spacy': ['spacy>=3.0.0'],
    },
    python_requires='>=3.9',
    author="Your Name",
    author_email="your.email@example.com",
    description="Research on LLM personality phase transitions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
) 