from gym.envs.registration import register

register(
    id='ToyLab-v0',
    entry_point='multi_goal.envs.toy_labyrinth_env:ToyLab',
)
