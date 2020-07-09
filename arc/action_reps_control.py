import os
import random
from itertools import count
from pathlib import Path
from typing import Sequence, Callable, List, Mapping, Optional

import gym
import numpy as np
import torch
from runstats import Statistics
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.her import HERGoalEnvWrapper
from torch import Tensor, nn

import torch.nn as nn
from torch.distributions import Distribution, MultivariateNormal as MVNormal
from torch.distributions.kl import kl_divergence as KL
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from multi_goal.GenerativeGoalLearning import Agent
from multi_goal.agents import HERSACAgent
from multi_goal.envs import Observation, ISettableGoalEnv, GoalHashable
from multi_goal.envs.toy_labyrinth_env import ToyLab

ObservationSeq = Sequence[Observation]
GaussianPolicy = Callable[[ObservationSeq], List[Distribution]]


def get_gaussian_pi(agent: HERSACAgent, env: ISettableGoalEnv) -> GaussianPolicy:
    flattener = HERGoalEnvWrapper(env)

    def gaussian_pi(many_obs: ObservationSeq) -> List[Distribution]:
        many_flat_obs = np.array([flattener.convert_dict_to_obs(o) for o in many_obs])
        mus, sigma_diags = agent._model.policy_tf.proba_step(many_flat_obs)
        t = Tensor
        return [MVNormal(t(mu), torch.diag(t(sig))) for mu, sig in zip(mus, sigma_diags)]

    return gaussian_pi


def copy_and_replace_goal(obss: ObservationSeq, goal: np.ndarray) -> ObservationSeq:
    new_obss = [Observation(**o) for o in obss]
    for obs in new_obss:
        obs["desired_goal"] = goal
    return new_obss


def Dact(g1: np.ndarray, g2: np.ndarray, D_: ObservationSeq, pi: GaussianPolicy) -> Tensor:
    obs_samples = random.sample(D_, k=7)
    N1s = pi(copy_and_replace_goal(obss=obs_samples, goal=g1))
    N2s = pi(copy_and_replace_goal(obss=obs_samples, goal=g2))
    kls = [KL(N1, N2) + KL(N2, N1) for N1, N2 in zip(N1s, N2s)]
    return torch.mean(Tensor(kls))


class ActionableRep(nn.Module):
    DEFAULT_ARC_DIM = 3

    def __init__(self, input_size: int = 2, seed=0):
        super().__init__()
        torch.random.manual_seed(seed)
        hidden_layer_size = 128
        self.arc_dim = self.DEFAULT_ARC_DIM

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, self.arc_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ARCDescentAgent(Agent):
    _fpath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _arc_fpath = os.path.join(_fpath, "pts/arc-bullet.pt")

    def __init__(self, env: ISettableGoalEnv, phi: ActionableRep = None, lr=0.8):
        if phi is None:
            self._phi = ActionableRep(input_size=env.observation_space["desired_goal"].shape[0])
            self._phi.load_state_dict(torch.load(self._arc_fpath))
        else:
            self._phi = phi
        self._x = torch.zeros(2, requires_grad=True)
        self._lr = lr
        self._opt = self._make_opt()

    def _make_opt(self):
        return torch.optim.Adam([self._x], lr=self._lr)

    def __call__(self, obs: Observation) -> np.ndarray:
        self._opt.zero_grad()
        cur_x = torch.from_numpy(obs.achieved_goal).float()
        self._x.data = cur_x.clone()
        goal = torch.from_numpy(obs.desired_goal).float()
        loss = torch.dist(self._phi(goal), self._phi(self._x))
        loss.backward()
        self._opt.step()
        direction = self._x - cur_x
        norming = max(1, torch.norm(direction))
        return (direction/norming).detach().numpy()

    def train(self, timesteps: int) -> None:
        raise NotImplementedError("The ARC agent is already trained.")

    def reset_momentum(self):
        if self._x.grad is not None:
            self._x.grad.zero_()
        self._opt = self._make_opt()


class ARCEnvWrapper(ISettableGoalEnv):
    def __init__(self, env: ISettableGoalEnv, phi: ActionableRep = None):
        self.__class__.__name__ = env.__class__.__name__
        self._delegate = env
        self._phi = phi if phi is not None else ActionableRep(input_size=env.observation_space["desired_goal"].shape[0])

        spaces = ["achieved_goal", "desired_goal"]
        random_states = {s: env.observation_space[s].np_random.get_state() for s in spaces}
        new_obs_dim = env.observation_space["observation"].shape[0] + 2*self._phi.arc_dim
        self.observation_space = gym.spaces.Dict(spaces={
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,)),
            "achieved_goal": env.observation_space["achieved_goal"],
            "desired_goal": env.observation_space["desired_goal"]
        })
        [env.observation_space[s].np_random.set_state(random_states[s]) for s in spaces]

        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.starting_agent_pos = env.starting_agent_pos
        self.max_episode_len = env.max_episode_len

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        return self._delegate.set_possible_goals(goals=goals, entire_space=entire_space)

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return self._delegate.get_successes_of_goals()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._delegate.compute_reward(achieved_goal=achieved_goal,
                                             desired_goal=desired_goal, info=info)

    def step(self, action):
        obs: Observation
        obs, r, done, info = self._delegate.step(action=action)
        obs = self._enrich_obs(obs)
        return obs, r, done, info

    def _enrich_obs(self, obs: Observation) -> Observation:
        rep = self._phi(Tensor([obs.achieved_goal, obs.desired_goal])).detach().numpy()
        new_obs = np.concatenate((obs.observation, *rep))
        return Observation(observation=new_obs, achieved_goal=obs.achieved_goal, desired_goal=obs.desired_goal)

    def render(self, *args, **kwargs):
        return self._delegate.render(*args, **kwargs)

    def reset(self) -> Observation:
        obs = self._delegate.reset()
        return self._enrich_obs(obs)

    def seed(self, seed=None):
        self._delegate.seed(seed)


class ARCTrainingAgent(Agent):
    def __init__(self, env: ISettableGoalEnv, rank=0, verbose=1):
        self._env = env
        self._phi = ActionableRep(input_size=env.observation_space["desired_goal"].shape[0], seed=rank)
        arc_env = ARCEnvWrapper(env=env, phi=self._phi)
        self._agent = HERSACAgent(env=arc_env, rank=rank, experiment_name="arc-her-sac", verbose=verbose)
        self._gaussian_pi = get_gaussian_pi(agent=self._agent, env=arc_env)
        self._replay_buffer: ReplayBuffer = self._agent._model.model.replay_buffer
        self._train_arc_callback = TrainARCCallback(train_fn=self._train_arc)
        self._rank = rank
        self._arc_tb_logger = SummaryWriter(log_dir=str(Path(self._agent._dirs.tensorboard)/f"ARC_{rank}"), )
        self._arc_step = count()
        self._dact_stats = Statistics([1.0])

    def __call__(self, obs: Observation) -> np.ndarray:
        return self._agent(obs)

    def train(self, timesteps: int, eval_env: ISettableGoalEnv = None) -> None:
        self._agent.train(timesteps=timesteps, callbacks=[self._train_arc_callback], eval_env=eval_env)
        torch.save(self._phi.state_dict(), f"arc-{self._rank}.pt")

    def _sample_transitions(self) -> ObservationSeq:
        num_samples = 100
        obss = self._replay_buffer.sample(num_samples)[0]
        obss = (self._agent._model.env.convert_obs_to_dict(o) for o in obss)
        return [Observation(**o) for o in obss]

    def _train_arc(self, current_step: int) -> None:
        dataset = self._sample_transitions()
        self._arc_tb_logger.add_scalar("ARC/Mean Dact", self._dact_stats.mean(), global_step=current_step)
        loop = train_arc(D=dataset, gaussian_pi=self._gaussian_pi, phi=self._phi, num_epochs=1, dact_stats=self._dact_stats)
        for loss, dact in loop:
            step = next(self._arc_step)
            self._arc_tb_logger.add_scalar("Loss/train", loss, global_step=step)
            self._arc_tb_logger.add_scalar("ARC/Dact Sample", dact, global_step=step)


class TrainARCCallback(BaseCallback):
    def __init__(self, train_fn: Callable[[int], None]):
        super().__init__()
        self._train_fn = train_fn

    def _on_step(self) -> bool:
        if self.num_timesteps != 0 and self.num_timesteps % 500 == 0:
            self._train_fn(self.num_timesteps)
        return True


def calc_avg_Dact(dataset: ObservationSeq, pi: GaussianPolicy, samples=50):
    def sample():
        o1, o2 = random.sample(dataset, 2)
        return Dact(o1.desired_goal, o2.desired_goal, dataset, pi=pi)

    return np.array([sample() for _ in range(samples)]).mean()


def train_arc(D: ObservationSeq, gaussian_pi: GaussianPolicy, phi: ActionableRep,
              num_epochs=10, dact_stats: Statistics = None):
    optimizer = AdamW(phi.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dact_stats = Statistics([1]) if dact_stats is None else dact_stats
    epoch_len = len(D)
    for epoch in range(1, num_epochs+1):
        losses = []
        for _ in range(epoch_len):
            obs1, obs2 = random.sample(D, k=2)
            s1, s2 = Tensor(obs1.achieved_goal), Tensor(obs2.achieved_goal)

            optimizer.zero_grad()
            dact = Dact(s1, s2, D, pi=gaussian_pi)
            loss = loss_fn(torch.dist(phi(s1), phi(s2)), dact / dact_stats.mean())
            loss.backward()
            optimizer.step()

            loss = float(loss)
            losses.append(loss)
            dact_stats.push(float(dact))
            yield loss, float(dact)
        print(f"Epoch finished: {epoch}, loss: {np.mean(losses):.3f}", flush=True)


if __name__ == '__main__':
    env = ToyLab(use_random_starting_pos=True)
    agent = ARCTrainingAgent(env=env, rank=1)
    agent.train(40000)
