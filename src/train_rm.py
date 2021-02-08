import os
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape

import numpy as np
import gym
import torch
import time
from rm_robot_env import RMRobotEnv
import argparse

import rlbench.gym
from spinup.algos.pytorch.ppo.ppo import ppo

# env = ReacherGymEnv({'headless': True, 'horizon': 100})


# nF = 7
# task_name = 'composite'
# nF = 5
# task_name = 'sequential'
# nF = 5
# task_name = 'IF'
# nF = 3
# task_name = 'OR'
epochs_per_state = 200
epoch_schedule =   [50, 100, 150, 200, 250, 1400]
horizon_schedule = [200, 600, 800, 900, 1000, 1200]


ac_kwargs = {
    'hidden_sizes': (128, 128, 128),
    # 'activation': torch.nn.Tanh
    'activation': torch.nn.LeakyReLU
}

# nFs = [5, 5, 3]
# task_names = ['sequential', 'IF', 'OR']
nFs = [3]
task_names = ['OR']
for nF, task_name in zip(nFs, task_names):
    logger_kwargs = {
        "output_dir": os.path.join(os.environ['PKG_PATH'], 'experiments', 'rm', task_name),
        "exp_name": "rm"
    }

    env = lambda : RMRobotEnv(nF=nF, task_name=task_name)

    ppo(env,
        logger_kwargs=logger_kwargs,
        ac_kwargs=ac_kwargs,
        clip_ratio=0.2,
        epochs=5000,
        pi_lr=3e-4,
        vf_lr=3e-4,
        lam=0.99,
        gamma=0.995,
        steps_per_epoch=3000,
        train_pi_iters=2,
        train_v_iters=5,
        target_kl=0.02,
        max_ep_len=1000,
        minibatch_size=256,
        log_gradients=False,
        save_freq=250
    )