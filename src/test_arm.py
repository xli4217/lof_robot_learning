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
import math
import pdb

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

from option import Subgoal

SCENE_FILE = join(dirname(abspath(__file__)), 'ttt',
                  'scene_reinforcement_learning_env.ttt')
#POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
sample_region_dim = np.array([0.3, 0.7, 0.4])
sample_region_pos = np.array([0.892,0, 0.961])
POS_MIN, POS_MAX = list(sample_region_pos - sample_region_dim/2), list(sample_region_pos + sample_region_dim/2) 


class RobotEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, headless=True, render_camera=False, option_rollout=False, episode_len=100):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Panda()
        
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        #self.gripper_sensor = ProximitySensor('Panda_gripper_grasp_sensor')
        self.option_rollout = option_rollout
        self.episode_len = episode_len
        
        self.gripper = PandaGripper()
        self.gripper_dummy = Dummy('Panda_tip')
        
        self.agent_config_tree = self.agent.get_configuration_tree()

        self.agent.set_joint_positions(self.agent.get_joint_positions())

if __name__ == '__main__':
    robenv = RobotEnv()
    print(robenv.agent.get_tip().get_position())
    print(robenv.agent.get_tip().get_orientation())
    print(robenv.agent.solve_ik([1, 0, 1.47], [-3.14159, 0.244, -3.14159]))
    
    path = robenv.agent.get_path(position=[1, 0, 1.47], euler=[-3.14159, 0.244, -3.14159])
    # path = robenv.agent.get_path(position=[1, 0, 1.1], euler=[0, 3.14159, 0])
    # path = robenv.agent.get_path(position=[0.98,0.14,1.19], euler=[0, math.radians(180), 0])
    # path = robenv.agent.get_path(position=[ 0.94999975, -0.30000001,  1.10000002], euler=[0, 3.14159, 0])
    print(path[-1]._path_points)

    pdb.set_trace()