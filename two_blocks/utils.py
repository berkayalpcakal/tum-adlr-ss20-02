import functools
import matplotlib.pyplot as plt
from two_blocks_env.collider_env import SettableGoalEnv
from two_blocks_env.toy_labyrinth_env import _denormalize


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


def render_goals_with_env(goals, plot, env: SettableGoalEnv):
    # Plotting
    data_to_plot = [_denormalize(g) for g in goals]
    if plot is None:
        plot = env._plot[0].get_axes()[0].scatter(*zip(*data_to_plot))
        plt.xlim((-13, 5))
        plt.ylim((-5, 5))
    else:
        plot.set_offsets(data_to_plot)
    env.render()
    return plot
