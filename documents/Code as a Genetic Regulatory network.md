

This is a realisation I came across the other night.

No matter how "robust" your code is, self-modifying a code base results in the imminent collapse of it. This is because, at heart, good code should have very few redundancies in the form of repeated code blocks. You should have the functions abstracted into callable scripts that are easy to implement in multiple locations. This is counter to the manner in which biological organisms function, we insert multiple redundancies in order to ensure, that if the worst occurs, we can still have a living organism capable of life. [[Modelling an LLM as a protein#^62d631]]

I believe there is a way to make a code base well-suited for self-modification that we desire in a self-optimising LLM system. 

The base requirements:

1. A framework capable of orchestrating generative models robustly, both their own parameters as well as the exact prompts, data handling, and scrapping they may need to do. (asyncflows)
2. A language that is well suited to probabilistic thoughts to help build appropriate world models (optional?)  https://arxiv.org/pdf/2306.12672.pdf https://cocolab.stanford.edu/papers/GoodmanEtAl2008-UncertaintyInArtificialIntelligence.pdfv
3. A method of evaluating the topology of the LLM through some set of flows (async flows and https://www.mdpi.com/1099-4300/24/5/622 , perhaps https://www.mdpi.com/1099-4300/24/5/735)
4. A method of modifying the topology of these networks in a phenotype preserving manner (sand boxing subsets of the network, choosing or duplicating a process that is topologically invariant in the flow)![[Pasted image 20240331131038.png]]
5. ![[Pasted image 20240331131051.png]]