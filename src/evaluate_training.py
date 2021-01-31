import gym
# from option import Option
from robot_env import RobotEnv
import os
import torch
from option import *
import numpy as np
from pathlib import Path
import contextlib 
import pdb

def make_subgoals_reacher(env):
    # name, prop_index, subgoal_index, state
    all_info = env.all_info
    red_target = Subgoal('red_target', 0, 0, all_info['red_target'])
    red_goal = Subgoal('red_goal', 1, 1, all_info['red_goal'])
    green_target = Subgoal('green_target', 2, 2, all_info['green_target'])
    green_goal = Subgoal('green_goal', 3, 3, all_info['green_goal'])
    blue_target = Subgoal('blue_target', 4, 4, all_info['blue_target'])
    blue_goal = Subgoal('blue_goal', 5, 5, all_info['blue_goal'])

    return [red_target, red_goal, green_target, green_goal, blue_target, blue_goal]

def make_taskspec_reacher():
    # go to R or G, then B, then Y, then R
    spec = 'F( (r|g) & F( b & F (y & F r)))'

    nF = 5
    nP = 10
    tm = np.zeros((nF, nF, nP))

    # S0
    #    r  g  b  y  e
    # 0  0  0  1  1  1
    # 1  1  1  0  0  0
    # 2  0  0  0  0  0
    # 3  0  0  0  0  0
    # G  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2:] = 1
    # S1
    #    r  g  b  y  e
    # 0  0  0  0  0  0
    # 1  1  1  0  1  1
    # 2  0  0  1  0  0
    # 3  0  0  0  0  0
    # G  0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    # S2
    #    r  g  b  y  e
    # 0  0  0  0  0  0
    # 1  0  0  0  0  0
    # 2  1  1  1  0  1
    # 3  0  0  0  1  0
    # G  0  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 3, 3] = 1
    tm[2, 2, 4] = 1
    # S3
    #    r  g  b  y  e
    # 0  0  0  0  0  0
    # 1  0  0  0  0  0
    # 2  0  0  0  0  0
    # 3  0  1  1  1  1
    # G  1  0  0  0  0
    tm[3, 4, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    # G
    #    r  g  b  y  e
    # 0  0  0  0  0  0
    # 1  0  0  0  0  0
    # 2  0  0  0  0  0
    # 3  0  0  0  0  0
    # G  1  1  1  1  1
    tm[4, 4, :] = 1

    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = []
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# composite task
# (F((r|g) & F(b & F y)) & G ! can) | (F((r|g) & F y) & F can) & G ! o
def make_taskspec_reacher_composite():
    # go to A or B, then C, then HOME, unless C is CANceled in which case just go to A or B then HOME
    spec = '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 7
    nP = 10
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch e
    # 0  0  0  1  1  0  0  0  0  0 1
    # 1  1  1  0  0  0  0  0  0  0 0
    # 2  0  0  0  0  1  0  0  1  1 0
    # 3  0  0  0  0  0  1  1  0  0 0
    # 4  0  0  0  0  0  0  0  0  0 0
    # 5  0  0  0  0  0  0  0  0  0 0
    # G  0  0  0  0  0  0  0  0  0 0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 2, 4] = 1
    tm[0, 3, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 2, 7] = 1
    tm[0, 2, 8] = 1
    tm[0, 0, 9] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  0  0  0  0  0  0  1
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  1  1  1  0  0
    # 4  0  0  0  1  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 3, 2] = 1
    tm[1, 4, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 3, 5] = 1
    tm[1, 3, 6] = 1
    tm[1, 3, 7] = 1
    tm[1, 6, 8] = 1
    tm[1, 1, 9] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  1  1  1  0  0  1  1  1
    # 3  1  1  0  0  0  1  1  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0
    tm[2, 3, 0] = 1
    tm[2, 3, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 3, 5] = 1
    tm[2, 3, 6] = 1
    tm[2, 2, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 2, 9] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  1
    # 4  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  1  0  0  0  0  1  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 6, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 6, 8] = 1
    tm[3, 3, 9] = 1
    # S4
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # 4  1  1  0  1  0  0  0  0  0  1
    # 5  0  0  1  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0
    tm[4, 4, 0] = 1
    tm[4, 4, 1] = 1
    tm[4, 5, 2] = 1
    tm[4, 4, 3] = 1
    tm[4, 6, 4] = 1
    tm[4, 6, 5] = 1
    tm[4, 6, 6] = 1
    tm[4, 6, 7] = 1
    tm[4, 6, 8] = 1
    tm[4, 4, 9] = 1
    # S5
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0
    # 5  1  1  1  0  0  0  0  0  0  1
    # G  0  0  0  1  1  1  1  1  1  0
    tm[5, 5, 0] = 1
    tm[5, 5, 1] = 1
    tm[5, 5, 2] = 1
    tm[5, 6, 3] = 1
    tm[5, 6, 4] = 1
    tm[5, 6, 5] = 1
    tm[5, 6, 6] = 1
    tm[5, 6, 7] = 1
    tm[5, 6, 8] = 1
    tm[5, 5, 9] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1
    tm[6, 6, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# IF task
# (F (a & F c) & G ! can) | (F a & F can)
def make_taskspec_reacher_if():
    # go to A, then B, then C, then HOME
    spec = '(F (r & F b) & G ! can) | (F r & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 5
    nP = 10
    tm = np.zeros((nF, nF, nP))

    # S0
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  1  0  1  0  0  0  0  0  1
    # 1  1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  0  1  1  1  0
    # G  0  0  0  0  0  1  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 3, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 3, 4] = 1
    tm[0, 4, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 3, 7] = 1
    tm[0, 3, 8] = 1
    tm[0, 0, 9] = 1
    # S1
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  0  0  0  0  0  1
    # 2  0  0  1  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 4, 4] = 1
    tm[1, 4, 5] = 1
    tm[1, 4, 6] = 1
    tm[1, 4, 7] = 1
    tm[1, 4, 8] = 1
    tm[1, 1, 9] = 1
    # S2
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  1  1  1  0  0  0  0  0  1
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  1  0  0  0  1  1  1  1  1  0
    tm[2, 4, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 4, 4] = 1
    tm[2, 4, 5] = 1
    tm[2, 4, 6] = 1
    tm[2, 4, 7] = 1
    tm[2, 4, 8] = 1
    tm[2, 2, 9] = 1
    # S3
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  1  1  1  1  0  1  1  1  1
    # G  1  0  0  0  0  1  0  0  0  0
    tm[3, 4, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 4, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 3, 8] = 1
    tm[3, 3, 9] = 1
    # G
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# OR task
# F ((a | b) & F c) & G ! o
def make_taskspec_reacher_or():
    # go to A, then B, then C, then HOME
    spec = 'F ((a | b) & F c) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 3
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  1  1  1  0  0  1  1  1
    # 1  1  1  0  0  0  1  1  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 1, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 0, 9] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  1  1  1  0  1  1
    # G  0  0  1  0  0  0  0  1  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 1, 6] = 1
    tm[1, 2, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 1, 9] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1
    tm[2, 2, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 0]

    safety_props = [4, 5, 6, 7, 8]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props


# sequential task
# F(a & F (b & (F c & F h))) & G ! o
def make_taskspec_reacher_sequential():
    # go to A, then B, then C, then HOME
    spec = 'F(a & F (b & (F c & F h))) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 5
    nP = 10
    tm = np.zeros((nF, nF, nP))

    # S0
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  1  1  1  1  0  1  1  1  1
    # 1  1  0  0  0  0  1  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 0, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 0, 9] = 1
    # S1
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  1  0  1  1  1  1  0  1  1  1
    # 2  0  1  0  0  0  0  1  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 2, 6] = 1
    tm[1, 1, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 1, 9] = 1
    # S2
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  1  1  0  1  1  1  1  0  1  1
    # 3  0  0  1  0  0  0  0  1  0  0
    # G  0  0  0  0  0  0  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    tm[2, 2, 6] = 1
    tm[2, 3, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 2, 9] = 1
    # S3
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  1
    # G  0  0  0  1  0  0  0  0  1  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 4, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 4, 8] = 1
    tm[3, 3, 9] = 1
    # G
    #    r  g  b  y  c cr cg cb cy  e
    # 0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props


def make_safetyspecs_reacher():
    return []

def save_dataset(exp_name, method_name, task_name, epoch, results):
    directory = Path(__file__).parent / 'dataset' / exp_name / method_name /task_name 
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(epoch) + '.npz'
    path_name = directory / file_name
    np.savez(path_name, results)

def test_all_epochs(metapolicy_class, metapolicy_name, task, task_name):
    # epochs = [i for i in range(0, 900, 50)] + [899]
    epochs = [3500]
    num_tests = 10
    for epoch in epochs:
        results = test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=num_tests)
        save_dataset('satisfaction', metapolicy_name, task_name, epoch, results)

def test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=10):
    # option_load_path = os.path.join(os.environ['LOF_PKG_PATH'], 'experiments',
    #     'red', 'pyt_save', 'model{}.pt'.format(epoch))
    option_load_path = os.path.join(os.environ['PKG_PATH'], 'src', 'model{}.pt'.format(epoch))

    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        # option = Option(option_load_path)
    pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
    place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
    pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
    place_green_option = Option(option_load_path, pick_or_place='place', target='green_goal')
    pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
    place_blue_option = Option(option_load_path, pick_or_place='place', target='blue_goal')
    
    options = [pick_red_option, place_red_option, pick_green_option,
               place_green_option, pick_blue_option, place_blue_option]

    print("MODEL {} | TESTING EPOCH {}, using model{}.pt".format(metapolicy_name, epoch, epoch))
    rewards, final_fsa_states, successes = run_rollout(task, options, num_tests, metapolicy_class)

    results = {'reward': rewards, 'epoch': epoch, 'success': successes, 'last_state': final_fsa_states}

    return results

###############
# Run Rollout #
###############
def run_rollout(task_spec, policies, num_episodes, metapolicy_class):
    _, safety_props = make_taskspec_reacher_composite()
    safety_specs = make_safetyspecs_reacher()
    # subgoals = make_subgoals_reacher(env)

    option_to_color = {0: 'r', 1: 'g', 2: 'b', 3: 'y'}
    goal_state = task_spec.nF - 1

    rewards = []
    final_fsa_states = []
    successes = []

    render_camera = False
    env = RobotEnv(headless=False, render_camera=render_camera, option_rollout=True, episode_len=300)

    for i in range(num_episodes):
        # env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
        subgoals = env.subgoals

        done = False
        task_done = False
        R = 0
        obs = env.reset()

        metapolicy = metapolicy_class(subgoals, task_spec, safety_props, safety_specs, env, policies)

        f = 0
        env_f = 0
        num_steps = 0
        prev_f = f
        success = False

        prev_option = -1
        stop_high_level = False
        terminated = False

        while not task_done:
            if not stop_high_level:
                option = metapolicy.get_option(env, f)
                print(f, option_to_color[option], env_f)
            if prev_option == option and terminated:
                stop_high_level = True
            prev_option = option
            prev_f = f
            while prev_f == f and not task_done:
                # env.render()

                a = policy.get_action(torch.from_numpy(obs).float())
                color = option_to_color[option]
                obs, reward, task_done, info = env.step(a, color=color)
                # print("FSA: {} | Goal: {} | reward {}".format(f, color, reward))
                env_f = info['f']
                prev_f = f
                f = metapolicy.get_fsa_state(env, f)
                R += reward

                if f == goal_state:
                    success = True
                    env.set_task_done(True)

                state = tuple(env.all_info['ee_p'])
                if metapolicy.is_terminated(env, state, option):
                    terminated = True
                    break
                else:
                    terminated = False

        rewards.append(R)
        final_fsa_states.append(f)
        successes.append(success)      
        print(f"Episode {i} return: {R} | FSA: {f}")

    env_can.close()
    env_not.close()
    return rewards, final_fsa_states, successes

#######
# Run #        
#######
def run_all_tests():
    task_spec_composite, _ = make_taskspec_reacher_composite()
    task_spec_if, _ = make_taskspec_reacher_if()
    task_spec_sequential, _ = make_taskspec_reacher_sequential()
    task_spec_or, _ = make_taskspec_reacher_or()

    # metapolicies = [ContinuousMetaPolicy, GreedyContinuousMetaPolicy, FSAQLearningContinuousMetaPolicy, FlatQLearningContinuousMetaPolicy]
    # metapolicy_names = [, 'greedy', 'fsa', 'flat']
    # tasks = [task_spec_composite, task_spec_sequential, task_spec_if, task_spec_or]
    # task_names = ['composite', 'sequential', 'IF', 'OR']
    metapolicies = [ContinuousMetaPolicy]
    metapolicy_names = ['lof']
    tasks = [task_spec_or]
    task_names = ['OR']

    for metapolicy, metapolicy_name in zip(metapolicies, metapolicy_names):
        for task, task_name in zip(tasks, task_names):
            print("TEST {} {}".format(metapolicy_name, task_name))
            test_all_epochs(metapolicy, metapolicy_name, task, task_name)

if __name__ == '__main__':
    run_all_tests()