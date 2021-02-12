import gym
# from option import Option
from rm_robot_env import RMRobotEnv
import os
import torch
from option import *
import numpy as np
from pathlib import Path
import contextlib 
import pdb

def save_dataset(exp_name, method_name, task_name, epoch, results):
    directory = Path(__file__).parent / 'dataset' / exp_name / method_name / task_name 
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(epoch) + '.npz'
    path_name = directory / file_name
    np.savez(path_name, results)

def test_all_epochs(metapolicy_name, nF, task_name):
    epochs = [i for i in range(0, 1500, 250)]
    # epochs = [0]
    num_tests = 2
    for epoch in epochs:
        results = test_epoch(epoch, nF, task_name, metapolicy_name, num_tests=num_tests)
        save_dataset('lunchbox', metapolicy_name, task_name, epoch, results)

def test_epoch(epoch, nF, task_name, metapolicy_name, num_tests=10):
    option_load_path = os.path.join(os.environ['PKG_PATH'], 'experiments',
        'rm', task_name, 'pyt_save', 'model{}.pt'.format(epoch))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        option = RMPolicy(option_load_path)

    print("MODEL {} | TESTING EPOCH {}".format(metapolicy_name, epoch))
    rewards, final_fsa_states, successes = run_rollout(nF, task_name, option, num_tests)

    results = {'reward': rewards, 'epoch': epoch, 'success': successes, 'last_state': final_fsa_states}

    return results

###############
# Run Rollout #
###############
def run_rollout(nF, task_name, policy, num_episodes):
    goal_state = nF - 1

    rewards = []
    final_fsa_states = []
    successes = []
    env = RMRobotEnv(nF=nF, task_name=task_name, headless=True, eval=True, episode_len=3000)

    for i in range(num_episodes):
        cancel = (i % 2) != 0

        done = False
        R = 0
        obs = env.reset(cancel=cancel)

        f = 0
        num_steps = 0
        prev_f = f
        success = False
        terminated = False

        while not done and R > -200:
            # env.render()
            cancel = False
            # cancel only on ODD episodes, and only when the FSA
            # transitions from the initial state to the next state
            if i % 2 == 1 and prev_f == 0 and f != 0:
                cancel = True
                print("CANCELLED")

            prev_f = f

            a = policy.get_action(torch.from_numpy(obs).float())
            obs, reward, done, info = env.step(a)
            # print("FSA: {} | Goal: {} | reward {}".format(f, color, reward))
            prev_f = f
            f = obs[0]
            R += reward

            if f == goal_state:
                success = True
                done = True

        rewards.append(R)
        final_fsa_states.append(f)
        successes.append(success)      
        print(f"Episode {i} return: {R} | FSA: {f}")

    env.shutdown()
    return rewards, final_fsa_states, successes

#######
# Run #        
#######
def run_all_tests():
    # nFs = [7, 5, 5, 3]
    # task_names = ['composite', 'sequential', 'IF', 'OR']
    nFs = [7]
    task_names = ['composite']

    for nF, task_name in zip(nFs, task_names):
        print("TEST {} {}".format('RM', task_name))
        test_all_epochs('RM', nF, task_name)

if __name__ == '__main__':
    run_all_tests()