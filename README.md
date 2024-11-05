![phase_analysis_free_energy_20241105_002022](https://github.com/user-attachments/assets/c2b0d90d-b65a-4733-ae32-3b2f32a0b30d)

Hey, this is a thing I made because I thought it would be cool. I wrote a couple documents about a year ago on my own notation and how I can see a bunch of relationships between how strings of prompts are generated and stuff I did for modeling intrinsically disordered proteins. 

Then I decided I wanted to see if I could make a LLM dream. So I made a structure for personalities:

```

@dataclass
class Memory:
    short_term: List[str]  # M_S - verbose context
    long_term: List[str]   # M_L - conversation summaries
    archival: List[str]    # M_A - archivist information

@dataclass
class Identity:
    goals: List[str]       # I_G - model goals
    methods: List[str]     # I_M - planned approaches
    self_image: str        # I_S - self perception
    world_view: str        # I_W - environment perception
    thoughts: List[str]    # I_T - general thoughts
```
and ran some tests on what could be derived from messing with those, temperature, and what they would generate. 

Make sure to install the requirements with:
```
pip install -r requirements.txt
```

The dream stuff isn't quite done yet, but you can run the phase stuff:

```
python -m flows.experiments.personality_phase_separation
```
Make sure to modify `default_experiment.json` and provide our OPENAI_API_KEY in a .env file so you can run the experiments. 

Then you can look at the plots by running:

```
python -m flows.visualization.visualize_cache
```
or by just going to the plots folder. 

My weird little thoughts are in the documents folder, feel free to poke around in them if you are interested but please don't judge me too hard for them. I didn't really ever think other people would read them, they are originally from an obsidian I made. 

The implementation of the thermodynamics stuff is a bit funky right now and could do with more investigation because while they seem reasonable I feel like I can find a better formulation for coherence and enthalphic binding in text. I'll probably need to do a difference where I create target answers and then calculate deltas between energy levels that way, but I don't have the time now so here ya go. 
