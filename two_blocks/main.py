import time
import numpy as np
from itertools import count

from two_blocks_env.collider_env import ColliderEnv, distance

env = ColliderEnv(visualize=True)
obs = env.reset()

new_goal = np.array([3, 3, 0.5]).reshape((3, 1))
print(f"desired goal: {new_goal.T}")

env.set_goal(new_goal)

done = False
action = env.action_space.sample()

for t in count():
    obs, reward, done, _ = env.step(action)
    time.sleep(1/240)

    if t % 50 == 0:
        print(f"achieved goal: {obs.achieved_goal.T},"
              f" distance to desired goal: {distance(obs.achieved_goal, obs.desired_goal)}")

    if done:
        print("SUCCESS!")
        break
