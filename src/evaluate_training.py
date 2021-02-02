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

# F ((a | b) & F c) & G ! o
def make_taskspec_lunchbox():
    # go to A, then B, then C, then HOME
    spec = 'F((rt | gt) & F(rg & F(bt & F rg))'

    # prop order:
    # rt rg gt gg bt bg e

    nF = 5
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  e
    # 0   0  1  0  1  1
    # 1   1  0  1  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # G   0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 1, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    # S1
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   1  0  1  1  1
    # 2   0  1  0  0  0
    # 3   0  0  0  0  0
    # G   0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    # S2
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   1  1  1  0  1
    # 3   0  0  0  1  0
    # G   0  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 3, 3] = 1
    tm[2, 2, 4] = 1
    # S3
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   0  0  0  0  0
    # 3   1  0  1  1  1
    # G   0  1  0  0  0
    tm[3, 3, 0] = 1
    tm[3, 4, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    # G
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # G   1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = []
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
    num_tests = 1
    for epoch in epochs:
        results = test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=num_tests)
        save_dataset('lunchbox', metapolicy_name, task_name, epoch, results)

def test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=10):
    # option_load_path = os.path.join(os.environ['LOF_PKG_PATH'], 'experiments',
    #     'red', 'pyt_save', 'model{}.pt'.format(epoch))
    option_load_path = os.path.join(os.environ['PKG_PATH'], 'src', 'model{}.pt'.format(epoch))

    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        # option = Option(option_load_path)
    pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
    place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
    pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
    # place_green_option = Option(option_load_path, pick_or_place='place', target='green_goal')
    pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
    # place_blue_option = Option(option_load_path, pick_or_place='place', target='blue_goal')
    
    options = [pick_red_option, place_red_option, pick_green_option,
               pick_blue_option]

    print("MODEL {} | TESTING EPOCH {}, using model{}.pt".format(metapolicy_name, epoch, epoch))
    rewards, final_fsa_states, successes = run_rollout(task, options, num_tests, metapolicy_class)

    results = {'reward': rewards, 'epoch': epoch, 'success': successes, 'last_state': final_fsa_states}

    return results

###############
# Run Rollout #
###############
def run_rollout(task_spec, policies, num_episodes, metapolicy_class):
    _, safety_props = make_taskspec_lunchbox()
    safety_specs = make_safetyspecs_reacher()
    # subgoals = make_subgoals_reacher(env)

    option_to_color = {0: 'rt', 1: 'rg', 2: 'gt', 3: 'gg', 4: 'bt', 5: 'bg'}
    goal_state = task_spec.nF - 1

    rewards = []
    final_fsa_states = []
    successes = []

    render_camera = False
    env = RobotEnv(headless=False, render_camera=render_camera, option_rollout=True, episode_len=300)

    for i in range(num_episodes):
        # env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
        subgoals = env.subgoals

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
            done = False
            if not stop_high_level:
                env.soft_reset()
                option = metapolicy.get_option(env, f)
                print(f, option, option_to_color[option], env_f)
            if prev_option == option and terminated:
                stop_high_level = True
            prev_option = option
            prev_f = f
            while not done:
                # env.render()

                a = policies[option].get_action(env.all_info) #torch.from_numpy(obs).float())
                # color = option_to_color[option]
                obs, reward, done, info = env.step(a, policies[option].get_target_name())
                # print(option, policies[option].get_target_name())
                # print("FSA: {} | Goal: {} | reward {}".format(f, color, reward))
                # env_f = info['f']
                prev_f = f
                f = metapolicy.get_fsa_state(env, f, info, policies[option].get_target_name())
                R += reward

                if f == goal_state:
                    success = True
                    task_done = True
                    done = True
                    # env.set_task_done(True)

                # state = tuple(env.agent.get_tip().get_position())
                if metapolicy.is_terminated(env, option):
                    terminated = True
                    break
                else:
                    terminated = False

        rewards.append(R)
        final_fsa_states.append(f)
        successes.append(success)      
        print(f"Episode {i} return: {R} | FSA: {f}")

    # env_can.close()
    # env_not.close()
    return rewards, final_fsa_states, successes

#######
# Run #        
#######
def run_all_tests():
    taskspec_lunchbox, _ = make_taskspec_lunchbox()

    # metapolicies = [ContinuousMetaPolicy, GreedyContinuousMetaPolicy, FSAQLearningContinuousMetaPolicy, FlatQLearningContinuousMetaPolicy]
    # metapolicy_names = [, 'greedy', 'fsa', 'flat']
    # tasks = [task_spec_composite, task_spec_sequential, task_spec_if, task_spec_or]
    # task_names = ['composite', 'sequential', 'IF', 'OR']
    metapolicies = [ContinuousMetaPolicy]
    metapolicy_names = ['lof']
    tasks = [taskspec_lunchbox]
    task_names = ['OR']

    for metapolicy, metapolicy_name in zip(metapolicies, metapolicy_names):
        for task, task_name in zip(tasks, task_names):
            print("TEST {} {}".format(metapolicy_name, task_name))
            test_all_epochs(metapolicy, metapolicy_name, task, task_name)

if __name__ == '__main__':
    run_all_tests()