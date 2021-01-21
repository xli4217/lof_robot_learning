"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
import os
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
import gym
from spinup.algos.pytorch.ppo.ppo import ppo
import torch
import time

SCENE_FILE = join(dirname(abspath(__file__)), 'ttt',
                  'scene_reinforcement_learning_env.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 5
EPISODE_LENGTH = 200


class RobotEnv(gym.Env):

    def __init__(self, headless=True):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        action_high = np.array(4 * [1.])
        #action_high[1] = -10.
        action_low = np.array(4 * [-1.])
        #action_low[1] = -20

        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array(17*[-10.]), high=np.array(17*[10.]), dtype=np.float32)
        self.current_step = 0
        
    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position        
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position()])

        # jp = np.array(self.agent.get_joint_positions())
        # jv = np.array(self.agent.get_joint_velocities())
        # ee_p = np.array(self.agent_ee_tip.get_position())
        # target_pos = np.array(self.target.get_position())

        # state = np.concatenate([np.cos(jp), np.sin(jp), target_pos, jv, ee_p-target_pos])

        return state
        
    def reset(self):
        # Get a random position within a cuboid and set the target position
        self.t_start = time.time()
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        done = False
        action = np.concatenate([action, np.zeros(3)])
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()

        # Reward is negative distance to target
        r_dist = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        r_action = - np.linalg.norm(action)
        
        reward = r_dist #+ 0.1*r_action
        reward *= 0.2
        if r_dist > -0.05:
            reward += 0.5
            done = True

        self.current_step += 1
        if self.current_step >= EPISODE_LENGTH:
            #print(f"episode time: {time.time()-self.t_start}")
            done = True
        info = {}
        
        return self._get_state(), reward, done, info

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()



if __name__ == "__main__":
        
    class Agent(object):
        
        def act(self, state):
            del state
            return list(np.random.uniform(-1.0, 1.0, size=(7,)))

        def learn(self, replay_buffer):
            del replay_buffer
            pass

        
    env = RobotEnv(headless=False)
    agent = Agent()
    replay_buffer = []

    EPISODES = 2
    for e in range(EPISODES):

        print('Starting episode %d' % e)
        state = env.reset()
        for i in range(EPISODE_LENGTH):
            #action = agent.act(state)
            action = np.array(7*[0])
            action[1] = -1.
            reward, next_state, _, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state))
            state = next_state
            agent.learn(replay_buffer)

    print('Done!')
    env.shutdown()
