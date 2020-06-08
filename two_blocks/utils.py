import functools
import matplotlib.pyplot as plt
from two_blocks_env.collider_env import SettableGoalEnv
from two_blocks_env.toy_labyrinth_env import _denormalize
import numpy as np

def print_message(msg: str):
    def decorator(func):
        @functools.wraps(func)
        def printer(*args, **kwargs):
            print(msg, end=" ", flush=True)
            res = func(*args, **kwargs)
            print("[DONE]")
            return res
        return printer
    return decorator


def render_goals_with_env(goals, returns, plot, env: SettableGoalEnv):
    # Plotting
    denormalized_goals = np.array([_denormalize(g) for g in goals])

    rewards = np.array(returns)
    low_reward_idx  = np.argwhere(0.1>rewards).reshape(-1,)
    high_reward_idx = np.argwhere(0.9<rewards).reshape(-1,)
    goid_reward_idx = np.argwhere(np.array([int(0.1 <= r <= 0.9) for r in returns])==1).reshape(-1,)

    low_reward_goals  = list(denormalized_goals[low_reward_idx] )
    high_reward_goals = list(denormalized_goals[high_reward_idx])
    goid_reward_goals = list(denormalized_goals[goid_reward_idx])

    if plot is None:
        plot = env._plot[0].get_axes()[0].scatter(*zip(*low_reward_goals), c=['red'])
        plot = env._plot[0].get_axes()[0].scatter(*zip(*high_reward_goals), c=['green'])
        plot = env._plot[0].get_axes()[0].scatter(*zip(*goid_reward_goals), c=['blue'])
#        plt.xlim((-13, 5))
#        plt.ylim((-5, 5))
    else:
        plot.set_offsets(list(denormalized_goals))
    env.render()
    return plot
