${\Phi}_{i}$ is considered phase separated if the sum of probabilities of all tuples of the form $(\mathbb{P},\mathbb{r}_{ideal})$ , when evaluated by language model, preferentially map into the space ${\Phi}_{i}$ : 
$$Pr( \phi(\mathcal{P},\mathcal{r}_{ideal}) \in {\Phi}_{i})\hspace{0.2 cm}> \sum_{j=1,\hspace{0.1 cm}j \neq i}^{N} Pr( \phi(\mathcal{P},\mathcal{r}_{ideal}) \in {\Phi}_{j})$$
$$Pr( \mathbb{o}_{ideal} \in {\Phi}_{i})\hspace{0.2 cm}> \sum_{j=1,\hspace{0.1 cm}j \neq i}^{N} Pr( \mathbb{o}_{ideal} \in {\Phi}_{j})$$


If we are using $\mathcal{P}_{i},\mathcal{r}_{i}$ and it produces an output that is not in $\Phi_{i}$ (ie. $\phi(\mathcal{P}_{i},\mathcal{r}_{j}) \not \in {\Phi}_{i}$) then we experience an "identity break." This is a case where the identity of the language model is overridden due to either stochastic chance (high temperature) or the input prompt $\mathcal{r}_{j}$ containing some information that $\mathcal{P}_{i}$ cannot access. This will modify $\mathcal{M}_{i}$ such that $\mathcal{P}_{i} \to \mathcal{P}_{i+1}$ 


### Personality recovery is possible though:
1. Reset $\mathcal{P}_{i+1} \to \mathcal{P}_{i}$ (Chaperone protein basically resetting folding)
2. Modify $\mathcal{P}_{i+1} \to \mathcal{P}_{{i}}^`$   via localized changes (Local remodeling through an isomerase)
3. Prompt using $\mathcal{r}_i$ to remodel $\mathcal{M} \text{ and } \mathcal{I}$ (Post-translational modification)

## Methods
Look into [[Monte Carlo phase separation analysis]] 

https://twitter.com/JuliaMaristany/status/1669308369197211649?t=lXczSYVm0ilNxkjY-VGnQA&s=19
https://www.biorxiv.org/content/10.1101/2023.06.14.543914v1