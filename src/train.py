import os
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape

import numpy as np
import gym
import torch
import time
from robot_env import RobotEnv
import argparse

import rlbench.gym
from spinup.algos.pytorch.ppo.ppo import ppo
#from spinup.algos.pytorch.ddpg.ddpg import ddpg
#from spinup.algos.pytorch.td3.td3 import td3


HEADLESS = True

logger_kwargs = {
    "output_dir": os.path.join(os.environ['PKG_PATH'], 'experiments', 'ppo_rlbench'),
    "exp_name": "ppo_rlbench"
}

ac_kwargs = {
    'hidden_sizes': (128, 128),
    'activation': torch.nn.Tanh
    #'activation': torch.nn.ReLU
}

# ddpg(lambda: RobotEnv(headless=HEADLESS),
#      logger_kwargs=logger_kwargs,
#      ac_kwargs=ac_kwargs,
#      max_ep_len=200,
#      epochs=300,
#      steps_per_epoch=3000
# )

# td3(lambda: RobotEnv(headless=HEADLESS),
#      logger_kwargs=logger_kwargs,
#      ac_kwargs=ac_kwargs,
#      max_ep_len=200,
#      epochs=300,
#      steps_per_epoch=3000
# )

env = lambda: RobotEnv(headless=HEADLESS)
#env = lambda: gym.make('reach_target-state-v0', render_mode='human')
ppo(env,
    logger_kwargs=logger_kwargs,
    ac_kwargs=ac_kwargs,
    clip_ratio=0.2,
    epochs=300,
    pi_lr=1e-4,
    vf_lr=2e-4,
    lam=0.99,
    steps_per_epoch=3000,
    train_pi_iters=10,
    train_v_iters=10,
    target_kl=0.01,
    max_ep_len=200
)

print('Done!')

