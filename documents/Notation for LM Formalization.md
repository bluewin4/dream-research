[[Personality matrix]]
[[Monte Carlo phase separation analysis]]

[[Soul]]

# Notation:
## Information Space 
$\mathbb{X}-$ 
	Objective informational space 
$\mathbb{X}_{\phi}-$ 
Subjective informational space accessed by language model $\phi()$, $\exists \hspace{0.1 cm} \Phi \in \mathbb{X}_{\phi}$ . Where $\phi(\mathbb{X}) \to \mathbb{X}_{\phi}$ . This mapping does not preserve the internal distances in $\mathbb{X}$.  
$\phi()-$ 
The language model prediction function generated from $\mathbb{X}_{\phi}$  
$\mathcal{\phi}^o()-$ 
The oracle that will perform the true transformation of input to output so as to access some $I_{target}$ 
$\phi^`()-$
	The decoding function, this function intakes some value and interprets. This can be in the form of a language model call, or in the form of some programmatic extraction.
${\Phi}$ - 
	The Personality Matrix, a subspace of $\mathbb{X}_{\phi}$ that is defined as the mapping for a set of personalities into $\mathbb{X}$ , the trivial case is where there is only 1 personality and no other input:
	$${\Phi}=\{\phi(\mathcal{P})\}=\mathcal{P} \mapsto \mathbb{X}$$
	$$\phi(\mathcal{P}) \hspace{0.1 cm} \circ \hspace{0.1 cm} \phi(\mathcal{P}) = \Phi_{2}(\mathcal{P})$$


$I-$ 
a piece of information, from how big a dog is to the name of your coworker. 
$\mathbb{I}=\{{I_{1},\dots,I_{n}}\}$ - Some set of information stored in $\mathbb{X}$
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
Guardrails functions using this logic, they are attempting to extract $I_{data}$ from $\mathcal{o}_{i}$. But they need a separation layer such that there is calculation of $\mathcal{r}_{I}$ from $\mathcal{r}_{i}$ . 

## Personality-[[Soul]]

$$\mathcal{P=[M,S,I]}$$
Wherein $\mathcal{P}$ is the personality of the language model that is defined by the values 

## [[Memory]]
$$\mathcal{M}=[M_{S},M_{L},M_{A}]$$$M_{S}$ - Shot term memory, verbose and full context. $r_{i}$ can be considered a subset of $M_{s}$. 
	$r_{i} \subseteq M_{s,i}$
$M_{L}$ - Long term memory, summaries of conversation topics
$M_{A}$ - Attentive/Archivist memory, information fed to the model by archivist ^9ce097

## Structure
$$\mathcal{S}=[S_{D},S_{T},S_{R}]$$
$S_{I}$ - Input, how the information being parsed is labeled (relevant for applications such as [[Corevax]])
$S_{T}$ - Tools the language model has available to use
$S_{O}$ - Output structure (eg. guardrails)

## Identity
$$\mathcal{I}=[I_{G},I_{M},I_{S}, I_{W}, I_{T}]$$
$I_{G}$ - Goals of the language model (I am doing G)
$I_{M}$ - Method/plan of language model (I will use M to do G_1)
$I_{S}$ - Self-image (I am S)
$I_W$ - Perception of the environment (The world is W)
$I_T$ - Thoughts on everything (I think that T)
 


## Detecting information in personality matrices
Assume:
$${\Phi},\mathbb{I} \subset \mathbb{X}_{\phi}$$

When ${\Phi}$ contains some subset of information $\mathbb{I}$ within it's bounds, then this creates the space:
$$\mathbb{I}_{i,{\Phi}}=\mathbb{I}_{i}  \cap  {\Phi}$$
When there is at least one member $I_{target} \in \mathbb{I_{\Phi}}_{,i}$ then it can be said that a personality matrix, ${\Phi}$, can access the information $\mathbb{I}_{i}$. 

First we recognize some $\Phi$ composed of some set LLM calls on a set of tuples $\{\phi(\mathcal{P}_{1} ,\mathcal{r}_{1}),..,\phi(\mathcal{P}_{i}, \mathcal{r}_{N}) \}$. Where the list contains n distinct $\mathcal{P}$ and N distinct $\mathcal{r}$. The order of operations is not necessarily direct.

$${\Phi} = \phi(\mathcal{P}_{1} ,\mathcal{r}_{1}) \to \phi(\mathcal{P}_{2} ,\mathcal{r}_{2}) \to \phi(\mathcal{P}_{1} ,\mathcal{r}_{3}) \to \dots \to\phi(\mathcal{P}_{n} ,\mathcal{r}_{j}) \to \phi(\mathcal{P}_{i} ,\mathcal{r}_{N})\to \mathcal{o}_{final}$$
$${\Phi}_{\mathcal{P}_{i}}=\{\phi(\mathcal{P}_{i}, \mathcal{r}_{j}) \text{ for any valid j=1,..,N} \}$$
${\Phi}_{\mathcal{P}_{i}}$ is essentially the space that a given sub-personality in a personality-matrix covers. This may be represented as some other set of logical/semantic meanings such as:
$${\Phi}_{\mathcal{P}_{i}}= \{\phi(\mathcal{P}_{i},  \mathcal{r}_{j}) \hspace{0.1 cm} \iff  j=\text{valid}\} \text{} \mapsto \mathbb{o}_{i} \hspace{0.1 cm}$$
$$
\mathcal{V}_{correct}(\mathcal{o}_{i})=\begin{dcases*}
\text{correct}, & \text{if $\exists \hspace{0.1 cm} I_{\Phi} \in \mathcal{o}_{i} $}\\
\text{incorrect}, & \text{if $ \exists \hspace{0.1 cm} I_{\Phi} \not \in 
\mathcal{o}_{i}$} 
\end{dcases*}$$
$$
\mathcal{V}_{valid}(\mathbb{o}_{i})=\begin{dcases*}
\text{valid}, & \text{if $\exists \hspace{0.1 cm} I_{\Phi} \in \mathbb{o}_{i} $}\\
\text{invalid}, & \text{if $\not \exists \hspace{0.1 cm} I_{\Phi} \in 
\mathbb{o}_{i}$} 
\end{dcases*}$$
and this can  be used to build a matrix of Boolean type values
^11b0ec

The information check $\mathcal{V}(\mathbb{o}_{i})$ can be used in [[Thermodynamics of LLM conformational space#Stability]]  
Important note: The temperature is very important to how much variation in response is garnered. As the process of mapping $\mathcal{P}$ onto $\mathbb{X}$ will become more stochastic.  ^287325

### Concise
A response is considered concise if it satisfies the following inequality: $$ Length(\phi_{concise}(\{ \mathcal{r}_{i},\mathcal{o}_{i} \}^{i=k}_{i=1})) < Length(\{ \mathcal{r}_{i},\mathcal{o}_{i} \}^{i=k}_{i=1})$$
Conciseness is closely related to the goal of [[Notation for LM Formalization#Compression]]. 

### Relevant 
When the following relation is true, then we consider a summarization of information to be relevant to the target information: $$ \frac{ \sum Dist(\forall \hspace{0.1 cm}I \in \mathbb{I}_{summarization}, I_{target})}{Size(\mathbb{I}_{summarization})} \leq  \frac{ \sum Dist(\forall \hspace{0.1 cm}I \in \mathbb{I}_{1\to k},I_{target})}{Size(\mathbb{I}_{1\to k})} $$
Where $Size()$ counts the number of entries in a set, and $Dist()$ is an arbitrary calculation on the distance between two pieces of information in $\mathbb{X}_{\Phi}$. Ideally this distance calculation will always be performed on $\mathbb{X}_{\phi}$ however frequently this is instead performed on some adjacent space

Relevance may be measured via the usage of some abstract usage of [[Personality Conformational Space sampling]] but generalized to information.

## Language model communication
[[Modelling an LLM as a protein#Understanding]]
What dictates "understanding" between two language models? $$\mathbb{I}_{i,j}={\Phi}_{i} \cap {\Phi}_{j} \in \mathbb{I}$$$$I_{communication} \in \mathcal{r}_{com}$$ such that 
		$$\phi^{`}(\phi(\mathcal{r}_{com},\mathcal{P}_{i}),\mathcal{P}_{j})=\phi^{\to`}(\mathcal{r}_{com},\{\mathcal{P}_{i},\mathcal{P}_{j}\})=I_{target}$$
		The question is:
		$$I_{communication} \stackrel{?}{=} I_{i,j}$$

In an ideal case this breaks down into:

$\text{The cat is blue.}=\mathcal{r}_{i}(\mathbb{I}_{{i}})$

The information conveyed by this sentence is:
$$\mathbb{I}_{i}=\{I_{cat},I_{color},I_{time} \}$$ **Note**: There is only one way the information is presented in the space for each entry so $\mathbb{I}_{cat}=I_{cat}$

The response searches a space with some model: $$\phi(\mathcal{r}_{i}(\mathbb{I}_{i}), \mathcal{P}_{j}) \mapsto \mathcal{o}_{i,j}(\mathbb{I}_{i,j})$$With both of these quantities being reliant on the context gained from the personality $\mathcal{P}_{j}$. Both the input $\mathcal{r}_{i}$ is colored by the glasses of perception and output $\mathcal{o}_{i,j}$ the glasses of utterance.

So if $$\mathcal{I}_{i}=\{I_{cat},I_{color}\}$$ with 
$I_{cat}=\{\text{Cats are gross and wet.}\}$
$I_{color}=\{\text{Blue is cold, but red is hot.}\}$


The output could be:

$\{\text{I bet it's cold and wet too}\}=\mathcal{o}_{i,j}(\mathbb{I}_{{i,j}})$ 

The information conveyed by this sentence is:
$$\mathbb{I}_{i,j}  \subset  \mathbb{I}_{i} \cap \mathcal{I}_{i}$$

Which is:

$\mathbb{I}_{i,j}=\{ I_{subject},I_{temperature},I_{wetness} \}$
with 

$I_{subject}=\{\text{It may have some properties}\}$
$I_{cold}=\{\text{is cold}\}$
$I_{wet}=\{\text{is wet}\}$

If $I_{target}=I_i \in \mathbb{I}_{i}$ 
	then for all possible $i$ we can say that $\not \exists I_{\mu}$, as 

And the hope for communication is when: 
$$\exists \hspace{0.1 cm } I_{target} \in \Phi^`(\mathbb{P}^`,\mathbb{r}^`) \cap
\Phi(\mathbb{P},\mathbb{r})$$
If, $I_{target} \in \{ \mathbb{I}_{i} \cap \mathbb{I}_{i,j} \}$ 
Then we consider $I_{target}=I_{\mu}$
I will define this state as when a language model is capable of communicating with another as " ${\Phi}_{\mathcal{P}_{i}}$ can communicate with ${\Phi}_{\mathcal{P}_{j}}^`$ on $\mathbb{I}_{target}$ " 

However this means there is still the concept of identifying where the information is in the chain that the overlap occurs. This is the overlap point wherein the two spaces can be used to convey information between one another. 



### Decoding information
The goal is for a decoding entity, $\phi^`()$, is to rederive $\mathcal{r}_{i}(\mathbb{I}_{i})$

The information available to a third party decoding entity, $\phi^`()$, is $\{\mathcal{r}_{i}(\mathbb{I}_{i}), \mathcal{o}_{ij}(\mathbb{I}_{i,j})\}$ 
This model has two types of incarnations: 
1. Where the model, $\phi^`()$, has access to, $\phi_{bias}^`( \mathcal{P}_{i})$ 
2. Where the model, $\phi^`()$, does not have access to $\mathcal{P}$, $\phi_{unbias}^`(\mathcal{P}_{j})$
3. Input visible $\phi^`(\mathcal{r}_{i}(\mathbb{I}_{i}))$
4. output visible $\phi^`(\mathcal{o_{i}}(\mathbb{I}_{i,j}))$

|                                     | $\mathcal{P}_{i}$                                                                     | $\mathcal{P}_{j}$                                                                       | $\mathcal{P}_{null}$                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| $\mathcal{r}_{i}(\mathbb{I}_{i})$   | $\mathcal{P}_{i}, \mathcal{r}_{i}(\mathbb{I}_{i}))$                                   | $\mathcal{P}_{j}, \mathcal{r_{i}}(\mathbb{I}_{i})$                                      | $\mathcal{P}_{null}, \mathcal{r}_{i}(\mathbb{I}_{i}))$                                   |
| $\mathcal{o}_{i}(\mathbb{I}_{i,j})$ | $\mathcal{P}_{i}, \mathcal{o_{i}}(\mathbb{I}_{i,j})$                                  | $\mathcal{P}_{j}, \mathcal{o_{i}}(\mathbb{I}_{i,j})$                                    | $\mathcal{P}_{null}, \mathcal{o_{i}}(\mathbb{I}_{i,j})$                                  |
| both                                | $\mathcal{P}_{i}, \mathcal{r}_{i}(\mathbb{I}_{i}), \mathcal{o_{i}}(\mathbb{I}_{i,j})$ | $\mathcal{P}_{j}, \mathcal{r}_{i}(\mathbb{I}_{i}), \mathcal{o_{i}}(\mathbb{I}_{i.j})$ | $\mathcal{P}_{null}, \mathcal{r}_{i}(\mathbb{I}_{i}), \mathcal{o_{i}}(\mathbb{I}_{i,j})$ |

This is the experiment to identify the most optimal workflow for pulling context on why a language model behaved as it did based on it's personality contribution. This tells us there is some ideal personality space:

Note: the addition of $\phi^o()$ denotes this is an oracle operation, ie. a perfect representation of getting from a $\to$ b. 

Assume that $I_{target}\in {\Phi}^o$

$$\Phi^o=\{\text{Ordered list with replacement of }\mathcal{P}_{1,..,n} \in \mathbb{P}_{i} \text{ such that } \exists \hspace{0.1 cm} I_{target} \in \mathcal{o}^o(\mathbb{I}_{f})  \}$$
We are attempting to cause a spontaneous [[Phase separation of personalities]]
The new decoders job is to [[Personalities dreams#^02e1ef]].
If there are n possible models $${\Phi}_{\mathcal{P}_{i}}=\{ \text{Some ordered list with replacement }\mathcal{P}_{1},..,\mathcal{P}_{f} \}$$
Then there is some low entropy model, with high enthalpy for something that will make me really solid outputs

We can constrain the number of steps in $\Phi$ to N and number of personalities $\mathcal{P}$ to n so that we only need nxN combinations

$$C^R(n,N)=\frac{(n+N-1)!}{N!(n-1)!}$$
We can demo some workflows that match [[Biomemetic computing]] structures of thought in order to prod towards the oracle. This is where this stuff will become more art than science it feels. The gentle prodding of the personality into a shape that suits our whims. We could use [[Evolutionary Prompt]] structures to identify the unique personalities that are most applicable to a situation. 
 
## Compression
How does a language model compress a document from a long number of letters to something much lower in tokens?






