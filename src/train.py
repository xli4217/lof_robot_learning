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
from robot_env import RobotEnv

logger_kwargs = {
    "output_dir": os.path.join(os.environ['PKG_PATH'], 'experiments', 'test'),
    "exp_name": "test"
}

ac_kwargs = {
    'hidden_sizes': (128, 128),
    'activation': torch.nn.Tanh
}

ppo(RobotEnv,
    logger_kwargs=logger_kwargs,
    ac_kwargs=ac_kwargs,
    clip_ratio=0.1,
    epochs=300,
    pi_lr=1e-4,
    vf_lr=1e-3,
    lam=0.99,
    steps_per_epoch=2000,
    train_pi_iters=40,
    train_v_iters=40,
    max_ep_len=200
)

print('Done!')
env.shutdown()
