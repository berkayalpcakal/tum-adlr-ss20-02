from arc.action_reps_control import ARCDescentAgent, ActionableRep, ARCTrainingAgent, \
    TrainARCCallback
from arc.solve_labyrinth import main
from multi_goal.envs.toy_labyrinth_env import ToyLab


def test_class_instantation():
    env = ToyLab()
    phi = ActionableRep()
    a = ARCDescentAgent(env=env, phi=phi)
    a2 = ARCTrainingAgent(env=env)
    cb = TrainARCCallback(phi=phi)
