Personality matrix *P* is a subspace of some larger information matrix *X* 
![[PXL_20230806_071147550.jpg]]
*P* $\subset$ X

Within *X* there is a vector field denoting a differential equation describing biased movement around the space. 

A personality matrix *P* is a subspace of *X* wherein the likelihood of passing on some information $Pr(I_p)> 1/2$ for any given input prompt. Essentially it's a region of the larger *X* where some information will persist and spread. A prion is a biological version of this, a replicating [[Prompt injections]] a language model version. 
![[PXL_20230806_073214724.jpg]]



![[PXL_20230806_073253697.jpg]]


We can simulate traversing theses spaces as similar to Markov chains on surfaces describing the solution to some differential equations (fluid dynamics)


![[PXL_20230806_074219262.jpg]]
![[PXL_20230806_074258684.jpg]]

Need to read: paper on why prompt injection is inevitable 

[[Prompt injections]] in this framework are: essentially the pushing of some system to a preferred state, but by doing so within *P_jailbreak* the energy level is much higher than it normally should be or the information area covered by P_jailbreak is not well protected by the RLHF.


When exposed to $I_p$ the language model will preferentially traverse some subspace of information and obey the local dynamics of that space *P*. This is useful as during fine tuning people attempt to make energy barriers between a user and restricted information. Shunting them to some other answer such as "I can't talk about that". The space *P* acts as a catalyst to lower the energy barrier between a prompt and a restricted response.

The copy number and spread can be simulated through contagion dynamics wherein a Markov chain is traversed similar to that "[[protect a sick friend]]" graph analysis method I made during covid


[[Phase separation of personalities]] 

Distillation into smaller models. Identifying the things that are required to keep around
https://www.marktechpost.com/2023/08/12/researchers-from-usc-and-microsoft-propose-universalner-a-new-ai-model-trained-with-targeted-distillation-recognizing-13k-entity-types-and-outperforming-chatgpts-ner-accuracy-by-9-f1-on-43-dataset/

Read meta gpt paper