import gym
from robot_env import RobotEnv
import os
import torch
import gym
from option import *
import numpy as np
from pathlib import Path
import contextlib 

# F((rt | gt) & F(rg & F(bt & F rg))
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


# F((bt) & F(rg & F((rt | gt) & F rg))
def make_taskspec_lunchbox2():
    # go to A, then B, then C, then HOME
    spec = 'F((bt) & F(rg & F((rt | gt) & F rg))'

    # prop order:
    # rt rg gt gg bt bg e

    nF = 5
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  e
    # 0   1  1  1  1  1
    # 1   0  0  0  1  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # G   0  0  0  0  0
    tm[0, 0, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 1, 3] = 1
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
    # 2   0  1  0  1  1
    # 3   1  0  1  0  0
    # G   0  0  0  0  0
    tm[2, 3, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 2, 3] = 1
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

# F((rt & F(rg & F(bt & F (rg & F (gt & F rg)))))
def make_taskspec_lunchbox3():
    # go to A, then B, then C, then HOME
    spec = 'F((rt & F(rg & F(bt & F (rg & F (gt & F rg)))))'

    # prop order:
    # rt rg gt gg bt bg e

    nF = 7
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  e
    # 0   0  1  1  1  1
    # 1   1  0  0  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # 4   0  0  0  0  0
    # 5   0  0  0  0  0
    # G   0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    # S1
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   1  0  1  1  1
    # 2   0  1  0  0  0
    # 3   0  0  0  0  0
    # 4   0  0  0  0  0
    # 5   0  0  0  0  0
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
    # 4   0  0  0  0  0
    # 5   0  0  0  0  0
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
    # 4   0  1  0  0  0
    # 5   0  0  0  0  0
    # G   0  0  0  0  0
    tm[3, 3, 0] = 1
    tm[3, 4, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    # S4
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # 4   1  0  1  1  1
    # 5   0  1  0  0  0
    # G   0  0  0  0  0
    tm[4, 4, 0] = 1
    tm[4, 4, 1] = 1
    tm[4, 5, 2] = 1
    tm[4, 4, 3] = 1
    tm[4, 4, 4] = 1
    # S5
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   0  0  0  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # 4   0  0  0  0  0
    # 5   1  0  1  1  1
    # G   0  1  0  0  0
    tm[5, 5, 0] = 1
    tm[5, 6, 1] = 1
    tm[5, 5, 2] = 1
    tm[5, 5, 3] = 1
    tm[5, 5, 4] = 1
    # G
    #    rt rg gt bt  e
    # 0   0  0  0  0  0
    # 1   0  0  0  0  0
    # 2   0  0  0  0  0
    # 3   0  0  0  0  0
    # 4   0  0  0  0  0
    # 5   0  0  0  0  0
    # G   1  1  1  1  1
    tm[6, 6, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 1, 1, 0]

    safety_props = []
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props


def make_safetyspecs_reacher():
    return []

def save_dataset(exp_name, method_name, task_name, test_num, results):
    directory = Path(__file__).parent / 'dataset' / exp_name / method_name /task_name 
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(test_num) + '.npz'
    path_name = directory / file_name
    np.savez(path_name, results)

def test_all_iters(metapolicy_class, metapolicy_name, task, task_name):
    num_tests = 1
    for i in range(num_tests):
        results = test_iters(metapolicy_class, task, metapolicy_name, i)
        save_dataset('composability', metapolicy_name, task_name, i, results)

def test_iters(metapolicy_class, task, metapolicy_name, test_num):
    iters = [i for i in range(1, 101, 10)] # [1] + [i for i in range(10, 100, 10)]

    option_load_path = os.path.join(os.environ['PKG_PATH'], 'experiments', 'ppo', 'pyt_save', 'model7000.pt')

    pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
    place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
    pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
    pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
    
    options = [pick_red_option, place_red_option, pick_green_option,
               pick_blue_option]

    # print("MODEL {} | TESTING EPOCH {}, using model{}.pt".format(metapolicy_name, epoch, epoch))
    rewards, final_fsa_states, successes = run_rollout(task, options, metapolicy_class, iters, test_num)

    results = {'reward': rewards, 'steps': iters, 'success': successes, 'last_state': final_fsa_states}

    return results

###############
# Run Rollout #
###############
def run_rollout(task_spec, policies, metapolicy_class, iters, test_num):
    _, safety_props = make_taskspec_lunchbox()
    safety_specs = make_safetyspecs_reacher()
    # subgoals = make_subgoals_reacher(env)

    option_to_color = {0: 'rt', 1: 'rg', 2: 'gt', 3: 'bt'}
    goal_state = task_spec.nF - 1

    rewards = []
    final_fsa_states = []
    successes = []

    render_camera = False
    env = RobotEnv(headless=True, render_camera=render_camera, option_rollout=True, episode_len=300)

    for i in iters:

        task_done = False
        obs = env.reset()
        subgoals = env.subgoals

        metapolicy = metapolicy_class(subgoals, task_spec, safety_props, safety_specs, env, policies,
                    num_hq_iter=i)

        f = 0
        R = 0
        env_f = 0
        num_steps = 0
        prev_f = f
        success = False

        prev_option = -1
        stop_high_level = False
        terminated = False

        while not task_done and R > -200 and num_steps < 50:
            print("EPISODE {} STEP {} REWARD {}##################".format(i, num_steps, R))
            num_steps += 1
            done = False
            if not stop_high_level:
                env.soft_reset()
                option = metapolicy.get_option(env, f)
                print(f, option_to_color[option], env_f)
            if prev_option == option and terminated:
                stop_high_level = True
            prev_option = option
            prev_f = f
            while not done:
                # env.render()

                a = policies[option].get_action(env.all_info)
                obs, reward, done, info = env.step(a, policies[option].get_target_name())
                # print("FSA: {} | Goal: {} | reward {}".format(f, color, reward))
                # env_f = info['f']
                prev_f = f
                f = metapolicy.get_fsa_state(env, f, info, policies[option].get_target_name())
                R += reward

                if f == goal_state:
                    success = True
                    task_done = True
                    done = True

                if metapolicy.is_terminated(env, option):
                    terminated = True
                    break
                else:
                    terminated = False

        if num_steps == 50:
            print("50 steps")
            R = -200

        rewards.append(R)
        final_fsa_states.append(f)
        successes.append(success)      
        print(f"Test Num: {test_num} | Iterations: {i} return: {R} | FSA: {f}")

    env.shutdown()
    return rewards, final_fsa_states, successes

#######
# Run #        
#######
def run_all_tests():
    task_spec_lunchbox, _ = make_taskspec_lunchbox()
    task_spec_lunchbox2, _ = make_taskspec_lunchbox2()
    task_spec_lunchbox3, _ = make_taskspec_lunchbox3()

    # metapolicies = [ContinuousMetaPolicy, GreedyContinuousMetaPolicy, FSAQLearningContinuousMetaPolicy]
    # metapolicy_names = ['lof', 'greedy', 'fsa']
    metapolicies = [ContinuousMetaPolicy, FSAQLearningContinuousMetaPolicy]
    metapolicy_names = ['lof', 'fsa']
    tasks = [task_spec_lunchbox, task_spec_lunchbox2, task_spec_lunchbox3]
    task_names = ['lunchbox', 'lunchbox2', 'lunchbox3']
    # metapolicies = [FSAQLearningContinuousMetaPolicy, FlatQLearningContinuousMetaPolicy]
    # metapolicy_names = ['fsa', 'flat']
    # metapolicies = [ContinuousMetaPolicy, GreedyContinuousMetaPolicy]
    # metapolicy_names = ['lof', 'greedy']
    # tasks = [task_spec_or]
    # task_names = ['OR']

    for metapolicy, metapolicy_name in zip(metapolicies, metapolicy_names):
        for task, task_name in zip(tasks, task_names):
            print("TEST {} {}".format(metapolicy_name, task_name))
            test_all_iters(metapolicy, metapolicy_name, task, task_name)

if __name__ == '__main__':
    run_all_tests()