from arc.action_reps_control import ARCAgent, ActionableRep
from arc.solve_labyrinth import main
from multi_goal.envs.toy_labyrinth_env import ToyLab


def test_class_instantation():
    env = ToyLab()
    a = ARCAgent(env=env)
    phi = ActionableRep()
