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
import argparse


HEADLESS = True

logger_kwargs = {
    "output_dir": os.path.join(os.environ['PKG_PATH'], 'experiments', 'test'),
    "exp_name": "test"
}

ac_kwargs = {
    'hidden_sizes': (64, 64, 64),
    'activation': torch.nn.Tanh
}


ppo(lambda: RobotEnv(headless=HEADLESS),
    logger_kwargs=logger_kwargs,
    ac_kwargs=ac_kwargs,
    clip_ratio=0.1,
    epochs=300,
    pi_lr=1e-4,
    vf_lr=1e-4,
    lam=0.99,
    steps_per_epoch=3000,
    train_pi_iters=40,
    train_v_iters=40,
    target_kl=0.01,
    max_ep_len=200
)

print('Done!')

