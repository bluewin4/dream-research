Memory is the formation of:
$$\mathcal{M}=[M_{S},M_{L},M_{A}]$$ $M_{S}$ - Shot term memory, verbose and full context
$M_{L}$ - Long term memory, summaries of conversation topics
$M_{A}$ - Attentive/Archivist memory, information fed to the model by archivist ^9ce097


## Short term memory
$M_{S}$ - Shot term memory, verbose and full context

For $k$ being the current numeration of the conversation and $n$ being the length of short term memory- 
$$M_{S,k}= \sum_{i=k}^{k-n} \{\mathcal{r}_{i},\mathcal{o}_{i} \}$$
$$M_{S,k} \subset_{i=k}^{i=k-n} \{\mathcal{r}_{i},\mathcal{o}_{i} \}$$
This may be modified such that only $\mathcal{r}_{i}$ or $\mathcal{o}_{i}$ is included in the output, in order to prevent redundant information. This is as $\mathcal{o}_{i} \mapsto \mathcal{r}_{i+1}$ quite neatly and it is expected that $\mathbb{I}_{i,j} \cup \mathbb{I}_{i=1} \approx \mathbb{I}_{i,j} \cap \mathbb{I}_{i=1}$ when performing a language model chain.

## Long term memory
$M_{L}$ - Long term memory, [[Notation for LM Formalization#Concise]]and [[Notation for LM Formalization#Relevant]] summary of a conversation.
Note: $\phi_{summarize}$ may be any arbitrary function to summarize, not just a language model call.

This is derived from some arbitrary summarization function upon the prior conversation. The exact algorithm and function for implementing this may vary. 
$$\mathcal{M}_{L,k}= \phi_{summarize}(\{[\mathcal{r}_{1},\mathcal{o}_{1}],\dots,[\mathcal{r}_{k}, \mathcal{o}_{k} ] \  \})$$

For cases where the summarization function produces a larger corpus than it's input than this is considered an expansion and the summarization function is defined as $\phi_{expand}$. This is a meta-cognitive effect that should be considered similar to a person's ability to narrativize some basic facts into a larger story that contains far more information. Which is a function that expands upon $I_{target} \in \mathbb{I}_{1\to k}$. 
$$\phi_{expand}(\mathbb{I}_{1\to k})=\mathcal{o}(\mathbb{I}_{expand})$$
Where
$$\mathbb{I}_{expand} \hspace{0.1 cm} \cap \mathbb{I}_{1\to k} = I_{target}$$


## Attentive Memory
$M_{A}$ - Attentive/Archivist memory, information fed to the model by [[Archivist]]

This is the set of information not present in  $\mathcal{S,M_{S},M_{L}}$ or $\mathcal{P}$ that is considered relevant to $I_{target}$.


In order to perform these operations a variety of options are available.