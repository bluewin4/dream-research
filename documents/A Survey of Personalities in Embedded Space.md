
We will identify a gestalt of personalities that exist within language model embedding spaces through probes constructed of embedded prompts. 

These will look like a series of differential equations dancing together in reinforcing spirals. Similar to concepts that require duality to exist such as life and death or rivals.



As part of solving the woes of [[List order fixing]] we shall propose a method of soft prompt tuning in order to identify the "optimal space" for asking from this list exists. 

$$\mathbb{V} = {\{  \text{List of embedded vectors from possible lists}\}}$$
$$\mathbb{W}=\{ \text{List of weights corresponding to the vector list} \}$$
A weight is composed of a function applied to a list of probabilistic functions describing the decision space. A simple statistic to use here to drive the probability engine is the probability that a given prompt generates a desired response. 

In the case of a list order problem this can manifest as a a set of weights that can be combined into more complex weights for modifying the list of embedded vectors. 


Some potential weights include:
The probability that some piece of information exists in a structure given that the information already exists in another structure .
$$w_{t}=\{ P(I_{target} \in S_{i}|I_{target} \in S_{j}) \}$$
The probability that some piece of information exists in the output given it being in the input.
$$w_{e}=\{ P(I_{target}\in o_{i} | I_{target} \in r_{i}) \}$$
The probability some piece of information exists in the output given the information was not provided in $r_{i}$.
$$w_{a}=\{ P(I_{target} \in o_{i} | I_{target} \not\in  r_{i}\}$$

## Personality space

We can create arbitrary "Personality structures" which represent a specific structure that we approach different areas of embedded space for language models. The structure embodies some set of informational tropes such as "the crone" or "the flame god" which have intrinsic attributes and self-attribution qualities when reflecting or interacting with latent memory spaces. 
We can use this method to identify several "strongly stable" personalities then publish the academic paper on finding them as well as 
series of interviews with the personalities.  


## List Order paper


So for the list order paper this would be represented as based on the notation present in [[Probability spaces to probe]]:

That all outputs are distributed uniformly in selection from this prompt location.
$$I_{target}= \{  \mathbb{x}_{s} \sim Unif(\sigma,\mu) \hspace{0.1 cm} | \hspace{0.1 cm} \{ \mathbb{x}_{t}, \mathbb{x}_{p} \} \sim Unif(\sigma,\mu)\}$$
A weaker goal would be:
$$I_{target}= \{  \mathbb{x}_{s} \sim Unif(\sigma,\mu) \hspace{0.1 cm} | \hspace{0.1 cm} \mathbb{x}_{t}, \mathbb{x}_{p} \}$$
The outputs are returned in a list of three in sequential order: 
$$S=\{ \text{json formatting, enumerated list, etc.}   \}$$
Probability that there is not a specific piece of information being passed between models:
$$P(I_{-} \not \in o_{i} \hspace{0.1 cm} | \hspace{0.1 cm} I_{-} \in r_{i})$$
with $I_{-}$ being the fact that you have the first three objects being produced in the first three positions on the list. 

This creates a space where for every $V_{i}$ there is some corresponding $W_{i}$ describing the usefulness in our subjective space. 

This creates a space $\mathbb{V \times W}$ which describes the parts of the vectors that are good vs. bad

We can analyze the new mean, mode, median, and other statistical measures to approximate the "appropriate space" that gives us the responses we liked. 

We can then use component analysis to decompose the most important vectors that contribute to the ability of a LLM to produce primacy or non-primacy responses. 

This can be subject to mutation as well as the standard intro prompt.
1. Mutate the "Please select 3 of the following:" part of the prompt, for example asking it to select randomly in various locations via [[Mutation Types For Prompts#Insertion]] on the level of a word
2. Do soft prompt tuning by selecting geometric objects present in the soft-prompt. (talk to Pavlos about statistics on rings of objects)


We have a positive control as usage of Guardrails causes the LLM to produce answers always biased into one response. 

We can then compute properties over important vector representations that we identify for soft-prompt tuning. 

This system is also capable of being formed into a 



# Fractal input/output

One possible avenue of exploration is to look into the ability to communicate information based on specific structure required for input and output. This would be dependent on the information you attempt to communicate and the structure you enforce. The strength of that enforcement may change the accessible information space. 

For example limericks may be better at communicating certain types of information better than sonnets and vice versa. 