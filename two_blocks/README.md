# Easy Environment
### Installation
```bash
python -m pip install -r requirements.txt
```
### Run
Requires python 3.6+
```bash
python main.py
```
Currently, there are two blocks that we can move using the mouse. Also, we can specify a goal. 
In `main.py`, we have specified that we want the blue block to reach the point
`[3, 3, 0.5]`, with the last coordinate being the height of the block on
the ground.

The environment is set up to apply the agent's action as force to the red block.
In `main.py` we currently take a (random) action and repeat it the whole episode.
If we move the blue block to the goal (e.g. with the mouse), the run ends successfully.