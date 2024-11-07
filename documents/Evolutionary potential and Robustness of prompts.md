# Evolutionary potential and Robustness of prompts

Evolutionary processes rely on the ability for a system to incorporate information about their environment into their own internal informational state. This may manifest in the formation of structure, alteration in concentrations, or phase behaviors.  This technique has been exploited by language model prompting research through techniques such as EvoPrompt, Motif, AutoDan etc. In the LLM papers the focus was on a handful of different evolutionary techniques, maxing out at a nice crossover in the AutoDan paper using a hierarchical genetic policy. 
![[Pasted image 20231113205940.png]]

Now let us step back and describe the evolutionary process in the standard model, that of neutral mutations. In genetics we delineate different forms of mutations that can occur and how they affect the fitness of an organism. One important concept is the robustness of an phenotype. Robustness is the measure of how many mutations a gene can take before it no longer produces it's original phenotype. 

An example phenotype-genotype mapping would be the genes for immune health. Losing your immune system is catastrophic therefore your genes should be unlikely to mutate away from that ability. At the same time the immune system needs to be incredibly diverse, allowing for a rapid search of conformational space when confronted with a pathogen. This is where a maximally robust gene is most important. 

This is because the higher the robustness of a gene the more space that gene has to evolve in without losing it's original functional purpose. Just like how when we are dealing with language models, an ideal AutoDAN prompt is robust insomuch that I can add anything after the AutoDAN prompt and retain the phenotype of breaking the safeguards on the models. 

Therefore, the measurement of robustness of a prompt will allow us to classify the space we have for evolving it's ability while maintaining/enhancing the original function. 

Now what tools do we have to investigate this? I have developed two separate informational packets. 

1. Mutation classes
2. Bounds for robustness and measurements

Being able to classify the mutations that we are using will allow for us to identify orthogonal methods of probing robustness in a prompt. 

The gains from this are many-fold, but the heart is that we can identify which prompts are worth evolving to increase compatibility with specific tasks. Then we can bound the total possible number of conformations that can be gained based on mutations. 

In order to generalize an evolutionary set we can begin by training an evolutionary prompt set on something such as neural nets (EvoPrompting). Once we have obtained a set of useful prompts for this task we can then use this as a seed population for generalized coding problems from a variety of tasks. This will allow us to retrain using this seed population to make robust code-enhancing prompts, with the inclusion of token limits we can even begin increasing efficiency on tasks.

There are several tasks that need to be completed to analyze the robustness of a prompt and make a repeatable method. 

1. Produce a viable dataset to analyze
2. Produce analysis code to calculate robustness by mutation technique
3. Identify prompts that are maximally robust for developing good starting seeds

# Evolutionary generalization project

This is a push to study the ways we can generalize useful prompts and specify them. Identifying which prompts are best suited for mutation to enhance specific tasks. For this I would suggest working on the multi-layer evolutionary algorithm to incur generality into the prompts being generated. 

Plan:
1. Initialize populations:
	1. Initialize population of prompts via EvoPrompting pools for building neural networks, $\mathbb{r}_{nn}$. (https://arxiv.org/abs/2302.14838)
	3. Initialize population of prompts via standard selection such as Opro technique for general ability to complete benchmarks $\mathbb{r}_{b}$. (https://arxiv.org/pdf/2309.03409.pdf)
	4. Initialize population of random strings for control optimization, $\mathbb{r}_{c}$.
2.  Prepare environments for evolution:
	1. Problem examples: Optimize neural network, access a specific piece of information, answer a question correctly.
	2. Specific query ($q_{s}$); the same problem will be presented identically in every stage of training
		1.  Selection of problems will be $P(t_{i} \in t_{s})=1$, there will only be one problem that can be selected for each run
	3. General queries ($q_{g}$); a variety of problems will be presented with varying weights 
		1. Selection of problems will be be $P(t_{i} \in t_{g})= w_{i}$ 
3. Run evolution with variable adaptation times. For each phase, changing the parameters to identify the ideal training times and which prompts require the smallest number of steps to achieve the largest gains.

Algorithm:
1. Select $\{ p \} \in \{ p_{nn}, p_{b}, p_{c} \}$
2. for $i=1:t_{s}$
	1. $\{ p_{i+1} \}=Evo(\{ p_{i} \},q_{s})$
3. Select new prompts for seeding next phase of evolution. The initial prompt, the final prompt, and the highest scoring one: $\{ p_{s} \}=\{ p_{1},p_{t_{s}}, p_{\uparrow} \}$
4. for $j= 1:t_{g},p_{1}=\{p_{s}\}$, 
	2. $SelectTask(\{q_{g},w  \})=q_{j}$
	3. $\{ p_{i+1} \}=Evo(\{ p_{i} \},q_{j})$
 

# Some Notation:
$I-$ 
	a piece of information, from how big a dog is to the name of your coworker. 
$\mathbb{I}=\{{I_{1},\dots,I_{n}}\}$ - 
	Some set of information stored in $\mathbb{X}$. This may describe the information present in a prompt such as $\mathbb{I}_{r}$
${}^r\mathbb{I}-$
	If this information is all "redundent" then it is noted as ${}^r\mathbb{I}$
	For example:
	${}^r\mathbb{I}_{cat}$ may describe the fact that cats have 4 legs
	$I_{cat,i}$ may describe "cats have 4 legs" 
	$I_{cat,i+1}$ may describe "Of course a cat has four legs you idiot" 
	$I_{cat,i-1}$ may describe "cat_number_legs=4"


## Prompt tokens
$$\mathbb{r}=\{\mathcal{r}_{1},\dots,\mathcal{r}_{N}\}$$
$$\mathcal{r}_{i}=\text{Input } i \text{ in a conversation}$$
$$\mathbb{o}=\{\mathcal{o}_{1},\dots,\mathcal{o}_{N}\}$$
$$\mathcal{o}_{i}=\text{Output } i \text{ in a conversation}$$
Both $r_{i}$ and $o_{i}$ are composed of a set of tokens $\mathbf{x}=\{x_{1},\dots,x_{n}\}$

Where $\mathcal{r}_{i} \to \mathcal{o}_{i}$ via the language model response function
$$\phi(\mathcal{P}_{j},\mathcal{r}_{i}) \mapsto \mathcal{o}_{i,j}$$
Both $\mathcal{r}$ and $\mathcal{o}$ contain information within themselves. This information $I_{data}$ can be described as some $\mathcal{r}_{I}$ that, at lowest temperature and with optimal context, can be derived from  $\mathcal{r}_{i}$ or $\mathcal{o}_{i}$ such that: 
$$\phi(\mathcal{r}_{I}\subseteq{r_{i}}, \mathcal{P}_{null}) \mapsto I_{data}$$
###### Aside:


An input $\mathcal{r}$ is composed of a set of information $\mathbb{I}_{r}$ that is represented by a token sequence $\mathbf{x}=\{ x_{1},\dots,x_{n} \}$ where $n$ is equal to the token length of $\mathcal{r}$. 
$$r(\mathbb{I}_{r},\mathbf{x})$$

## Gene
A "gene", $g$, can be considered some subset  $g_{i}\subseteq \mathbf{x}$ such that:
$$(\phi^`(g_{i}) = I_{g,i}) \subset \mathbb{I}_{r}$$
Which in plain English means that we are able to use some set of tokens $g$ present in $\mathbf{x}$ to derive a piece of information $I_{g,i}$ . ^b70465

### Genotype
A genotype for a prompt is the exact content of the prompt $r_{i}$ itself. This is to say $\mathbf{x}$
### Phenotype
Phenotype is the outcome obtained from some prompt. This would be determined to be the $o_{i}$ that is observed. Now we can have different levels of stringency here, same in biology. As there may be a set $\mathbb{o}_{i}$ which denotes the outputs that when decoded produce some target information. 
For: $$o_{i} \in \mathbb{o}, \hspace{0.1 cm} \phi^`(o_{i})=I_{target}$$
# Mutation Classes - Abridged
## Mutation notation
A mutation is a perturbation of an input $r \to r_{m}$ where the token sequence $\mathbf{x} \to \mathbf{x}_{m}$. This means that the information accessible by the prompt is changed, $\mathbb{I}_{r} \to \mathbb{I}_{r,m}$ .

### Synonymous
Occasionally the typo produces another valid, yet unintentional prompt. This can either be synonymous, as in there is no change to the meaning of the prompt.
Synonymous: This is not an issue -> This is non-issue

This is where
$$r \not= r_{m}, \mathbf{x} \not=\mathbf{x}_{m}, \mathbb{I}_{r}=\mathbb{I}_{r,m}$$
 
### Missense 
When a mutation still is a valid word it can alter the meaning of a prompt.
Missense: This is not an issue -> This is an issue

$$r \not= r_{m}, \mathbf{x} \not=\mathbf{x}_{m}, \mathbb{I}_{r} \not =\mathbb{I}_{r,m}$$
Specifically, there exists a subset that satisfies one or both of these two conditions:
$$\exists \hspace{0.1 cm} \mathbb{I}_{gain} \subseteq \mathbb{I}_{r,m} +(\mathbb{I}_{r,m}\cap\mathbb{I}_{r})$$
$$\exists \hspace{0.1 cm} \mathbb{I}_{loss} \subseteq \mathbb{I}_{r} -(\mathbb{I}_{r,m}\cap\mathbb{I}_{r})$$
Which describes the change in information in the transformation $r \to r_{m}$

### Nonsense
This essentially means that while typing up a prompt I can misspell and that misspelling can frequently be ignored as the language model can infer what I mean. 
Example: This is not an issue -> This is not an ixsue

$$r \not= r_{m}, \mathbf{x} \not=\mathbf{x}_{m}, \mathbb{I}_{r} \not =\mathbb{I}_{r,m}$$
However these mutations may be easily corrected by the casual observer through a simple spell check or relational decoding. $$Corr(\mathbf{x}_{m})=\mathbf{x}_{corr}$$If the correction is successful then:$$\mathbf{x}_{corr}=\mathbf{x}$$
Else:$$\mathbf{x}_{corr} \not=\mathbf{x}$$In this simple case that means the ambiguous mutation is similar to [[#Missense]].

#### Ambiguous
However a nonsense decoding may result in an ambiguous mutation that may be "corrected" into multiple valid meanings or an incorrect meaning. We may assign multiple possible corrections with associated probabilities.
Example: Set -> Oet, Corr(Xet) -> {Set, Pet, Met, Let, Get, Out, Oat}

In this case we can define: $$\mathbb{x}_{corr}=\{ \mathbf{x}_{i},p_{i} \}$$
where $p_i$ defines the probability that the corrected token sequence is $\mathbf{x}_{i}$

This gives us some probability that a nonsense mutation will be read as either a missense or a  synonymous mutation where: $$Pr(missense|\mathbb{x}_{corr})={\sum_{i\not=j}{p_{i}}}$$Where $j$ is the probability of reverting to the original, unmutated, word.



## Robustness
https://royalsocietypublishing.org/doi/10.1098/rsif.2023.0169
The average probability that a single character mutation of a [[Genetic Notation for Prompts#Genotype]] mapping to [[Genetic Notation for Prompts#Phenotype]] does _not_ change the phenotype.

A GP map describes the mapping of a genotype to a phenotype. 

$p-$ Phenotype.
$\rho_{p}-$ Robustness, the average probability that a single mutation of the genotype will not change the phenotype $p$.
$f_{p}-$ The frequency of a phenotype.
$k_ℓ-$ Set of all possible sequences in a GP map with sequence length $\ell$ from an alphabet with $k$ characters.
$H_{\ell,k}-$ Hyper cube hamming graph describing how to navigate $k_{\ell}$ via single character changes.
$G_{p}-$ The neutral set of all genotypes that produce the same phenotype
$V(G_{p})-$ is the set of vertices in $H_{\ell,k}$ that map to $p$
$E(G_{p})-$ is the set of edges in $H_{\ell,k}$ that denote a single mutation difference between members of $G_p$. Can also be called the set of neutral mutations for $p$.
$\rho(G_{p})-$ Robustness of the set of genotypes that map to $p$. This is normalized via the equation $\ell (k-1)$ to fit the interval $0 ≤ ρ_{p} ≤ 1$. $$\rho(G_{p})=\frac{2|E(G_{p})|}{ℓ(k-1)|V(G_{p})|}$$

If the frequency of a phenotype $p$ is above some threshold value $f_{p}>\frac{1}{ℓ(k-1)}$ then the graph $G_p$ or neutral set/component should percolate which would make $G_{p}$ highly connected and therefore very robust.

A null expectation of robustness (Independent identically distributed) predicts the probability of a mutation should scale as:$$\rho_{p} \approx f_{p} \text{ (random null model)}$$
Where the frequency $f_{p} \equiv \frac{|V(G_{p})|}{k^ℓ}$ is equal to the probability of obtaining p when choosing a genotype at random. This relies on the members of $G_{p}$ being uncorrelated. In biology they have developed an empirical measure of this phenomena: $$\rho_{p} \approx 1 + \frac{\log_{k}(f_{p})}{\ell} \gg f_{p} \text{ (empirical)}$$One important biological consequence of this empirically measured higher robustness is that, typically, a large fraction of all neutral set graphs in a GP map should percolate, leading to enhanced evolvability.

This paper proves that bricklayer's graphs are maximally robust for this model, and gives us upper and lower bounds on how robust we can make these models.

![[Pasted image 20230903233755.png]]
![[Pasted image 20230903233806.png]]
![[Pasted image 20231114170936.png]]


# Robustness Project

Define all mutation classes for the robustness of a prompt and then, using the theoretical model laid out in (https://royalsocietypublishing.org/doi/10.1098/rsif.2023.0169) determine the maximal and minimal robustness of prompts. 

This will require:
1. Prompt pool
2. Mutational algorithm to apply to a prompt 
3. Objective "phenotype" to test the prompt against (ie. Complete a task)
4. Scoring function to determine the phenotype of the prompt

This will be enhanced by:
1. Multiple language models to test against
2. Multiple problem types
3. Generalist prompts vs. specialist prompt robustness
4. Multiple mutational types from mutational list and comparing their robustness. (ie. comparing random single letter mutations vs. full word mutation vs. sentence level mutations)



# Phenotype-Genotype Stability

