import math
import time
from itertools import repeat, chain, count

import gym
import numpy as np
from pybullet_robots.panda.panda_sim_grasp import PandaSim
from pybullet_robots.panda.panda_sim_grasp import ll, ul, jr, rp, pandaEndEffectorIndex, pandaNumDofs
from pybullet_utils.bullet_client import BulletClient
import pybullet_data as pd
import pybullet

from multi_goal.envs import Simulator, SimObs, SettableGoalEnv


class PandaEnv(SettableGoalEnv):
    def __init__(self, visualize=False, seed=0, max_episode_len=200, use_random_starting_pos=False):
        super().__init__(sim=PandaSimulator(visualize=visualize), max_episode_len=max_episode_len, seed=seed,
                         use_random_starting_pos=use_random_starting_pos)


class PandaSimulator(Simulator):
    _num_joints = 12
    _arm_joints = list(range(pandaNumDofs))
    _grasp_joints = [9, 10]

    def __init__(self, visualize=False):
        self._p = p = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, -9.8, 0)
        p.loadURDF("plane.urdf", baseOrientation=p.getQuaternionFromEuler([-math.pi/2, 0, 0]))
        self._pandasim = PandaSim(bullet_client=p, offset=[0, 0, 0])
        self._remove_unnecessary_objects()

        abs_lego_starting_pos = [0, 0.015, -0.5]
        abs_lego_starting_euler_orn = [-np.pi/2, 0, 0]
        p.resetBasePositionAndOrientation(self._pandasim.legos[0], abs_lego_starting_pos, p.getQuaternionFromEuler(abs_lego_starting_euler_orn))
        self.normed_starting_agent_obs = self._get_all_legos_pos_and_orns()

        self._original_joint_states = self._p.getJointStates(self._pandasim.panda, range(self._num_joints))
        self._goal_pos = None

        goal_space = gym.spaces.Box(low=np.array([-1, 0, -1] + [-np.inf]*4),
                                    high=np.array([1]*3 + [np.inf]*4))  # lego: 3 pos + 4 orn
        self.observation_space = gym.spaces.Dict(spaces={
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2*self._num_joints, )),  # 2 = pos + vel
            "desired_goal": goal_space,
            "achieved_goal": goal_space
        })
        self.action_dim = 4  # [dx, dy, dz, dgrasp] as a change in gripper 3d-positon, where the y-axis references the height"""
        self._sim_steps_per_timestep = 20
        self._do_visualize = visualize

    def _remove_unnecessary_objects(self):
        spheres_ids = [5, 6, 7]
        legos_ids = [3, 4]
        tray_id = [1]
        [self._p.removeBody(e) for e in spheres_ids + legos_ids + tray_id]

    def step(self, action: np.ndarray) -> SimObs:
        action = [*np.array(action)[:3] / 100, action[-1]]
        cur_pos, *_ = self._p.getLinkState(self._pandasim.panda, pandaEndEffectorIndex)
        orn = self._p.getQuaternionFromEuler([math.pi/2., 0., 0.])
        pos = [coord+delta for coord, delta in zip(cur_pos, action)]

        grasp_forces = [10, 10]
        dgrasp = action[-1]

        joint_nums = chain(self._arm_joints, self._grasp_joints)
        forces = chain(repeat(5*240, pandaNumDofs), grasp_forces)
        desired_poses = self._p.calculateInverseKinematics(self._pandasim.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
        desired_poses = chain(desired_poses[:-2], [dgrasp, dgrasp])

        for _ in range(self._sim_steps_per_timestep):
            for joint, pose, force in zip(joint_nums, desired_poses, forces):
                self._p.setJointMotorControl2(self._pandasim.panda, joint, self._p.POSITION_CONTROL, pose, force=force)
            self._p.stepSimulation()
            if self._do_visualize:
                time.sleep(1/240)

        joint_pos_and_vels = self._get_joints_info()
        legos_pos_and_orns = self._get_all_legos_pos_and_orns()
        return SimObs(agent_pos=legos_pos_and_orns, obs=joint_pos_and_vels, image=np.empty(0))

    def _get_all_legos_pos_and_orns(self) -> np.ndarray:
        pos_and_orns = [self._p.getBasePositionAndOrientation(l) for l in self._pandasim.legos[:1]]  # Just the first lego for now.
        return np.array([num for (pos, orn) in pos_and_orns for num in chain(pos, orn)])

    def _get_joints_info(self) -> np.ndarray:
        needed_indices = list(chain(self._arm_joints, self._grasp_joints))
        joint_states = self._p.getJointStates(self._pandasim.panda, range(self._num_joints))
        pos, vels, *_ = zip(*np.array(joint_states)[needed_indices])
        return np.array([*pos, *vels])

    def set_agent_pos(self, pos: np.ndarray) -> None:
        self._reset_panda_to_original_joint_states()
        self._p.resetBasePositionAndOrientation(self._pandasim.legos[0], pos[:3], pos[3:])

    def set_goal_pos(self, pos: np.ndarray) -> None:
        self._goal_pos = pos

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        achieved_pos = achieved[:3]
        desired_pos = desired[:3]
        dist = np.linalg.norm(np.subtract(achieved_pos, desired_pos))
        return dist < 0.01

    def render(self, *args, **kwargs):
        pass

    def _reset_panda_to_original_joint_states(self):
        joint_pos_and_vels = [(state[0], state[1]) for state in self._original_joint_states]
        for idx, (pos, vel) in enumerate(joint_pos_and_vels):
            self._p.resetJointState(self._pandasim.panda, idx, pos, vel)


def keyboard_control():
    """Returns an action based on key control.
    horizontal: {w,a,s,d} vertical: {y,x},  gripper: {c}"""
    cmd = input("press one key to move: {w,a,s,d,x,y,c}")
    red, green, blue, grip = 0, 0, 0, 0  # Axis colors
    if cmd == 'w':
        red += 1
    if cmd == 's':
        red -= 1
    if cmd == 'a':
        blue -= 1
    if cmd == 'd':
        blue += 1
    if cmd == 'y':
        green -= 1
    if cmd == 'x':
        green += 1
    if cmd == 'c':
        grip += 1
    return np.array([red, green, blue, grip])


if __name__ == '__main__':
    env = PandaEnv(visualize=True)
    while True:
        obs = env.reset()
        for t in count():
            action = np.random.uniform(-1, 1, 4)  # keyboard_control()
            obs = env.step(action=action)
            if t >= 200:
                break