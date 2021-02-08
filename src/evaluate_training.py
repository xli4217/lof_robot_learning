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
def make_taskspec_or():
    # go to A, then B, then C, then HOME
    spec = 'F((rt | gt) & F rg)'

    # prop order:
    # rt rg gt gg bt bg e

    nF = 3
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  c  e
    # 0   0  1  0  1  1  1
    # 1   1  0  1  0  0  0
    # G   0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 1, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 0, 5] = 1
    # S1
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   1  0  1  1  1  1
    # G   0  1  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    # G
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # G   1  1  1  1  1  1
    tm[2, 2, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 0]

    safety_props = []
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# F ((a | b) & F c) & G ! o
def make_taskspec_if():
    # go to A, then B, then C, then HOME
    spec = '(F (rt & F rg) & G ! can) | (F rt & F can) & G ! o'

    # prop order:
    # rt rg gt gg bt bg e

    nF = 4
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  c  e
    # 0   0  1  1  1  0  1
    # 1   1  0  0  0  0  0
    # 2   0  0  0  0  1  0
    # G   0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 2, 4] = 1
    tm[0, 0, 5] = 1
    # S1
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   1  0  1  1  0  1
    # 2   0  0  0  0  0  0
    # G   0  1  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 3, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 1, 5] = 1
    # S2
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   1  0  1  1  1  1
    # G   0  1  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 3, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    # S3
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # G   0  1  0  0  0  0
    tm[3, 3, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 0]

    safety_props = []
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# F ((a | b) & F c) & G ! o
def make_taskspec_sequential():
    # go to A, then B, then C, then HOME
    spec = 'F (rt & F (gt & F (bt & F (rg))))'

    # prop order:
    # rt rg gt gg bt bg c e

    nF = 5
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  c  e
    # 0   0  1  1  1  1  1
    # 1   1  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   0  0  0  0  0  0
    # G   0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 0, 5] = 1
    # S1
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   1  1  0  1  1  1
    # 2   0  0  1  0  0  0
    # 3   0  0  0  0  0  0
    # G   0  0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    # S2
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   1  1  1  0  1  1
    # 3   0  0  0  1  0  0
    # G   0  0  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 3, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    # S3
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   1  0  1  1  1  1
    # G   0  1  0  0  0  0
    tm[3, 3, 0] = 1
    tm[3, 4, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    # G
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   0  0  0  0  0  0
    # G   1  1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = []
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# F ((a | b) & F c) & G ! o
def make_taskspec_composite():
    # go to A, then B, then C, then HOME
    spec = '(F ((rt | gt) & F (bt & F rg)) & G !can) | F((rt|gt) & F rg) & F can)'

    # prop order:
    # rt rg gt gg bt bg c e

    nF = 7
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # S0
    #    rt rg gt bt  c  e
    # 0   0  1  0  1  0  1
    # 1   1  0  1  0  0  0
    # 2   0  0  0  0  1  0
    # 3   0  0  0  0  0  0
    # 4   0  0  0  0  0  0
    # 5   0  0  0  0  0  0
    # G   0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 1, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 2, 4] = 1
    tm[0, 0, 5] = 1
    # S1
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   1  0  1  0  0  1
    # 2   0  0  0  0  0  0
    # 3   0  0  0  1  1  0
    # 4   0  1  0  0  0  0
    # 5   0  0  0  0  0  0
    # G   0  0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 4, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 3, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 1, 5] = 1
    # S2
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  1  0  1  1  1
    # 3   1  0  1  0  0  0
    # 4   0  0  0  0  0  0
    # 5   0  0  0  0  0  0
    # G   0  0  0  0  0  0
    tm[2, 3, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    # S3
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   1  0  1  1  1  1
    # 4   0  0  0  0  0  0
    # 5   0  0  0  0  0  0
    # G   0  1  0  0  0  0
    tm[3, 3, 0] = 1
    tm[3, 6, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    # S4
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   0  0  0  0  0  0
    # 4   1  1  1  0  0  1
    # 5   0  0  0  1  0  0
    # G   0  0  0  0  1  0
    tm[4, 4, 0] = 1
    tm[4, 4, 1] = 1
    tm[4, 4, 2] = 1
    tm[4, 5, 3] = 1
    tm[4, 6, 4] = 1
    tm[4, 4, 5] = 1
    # S5
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   0  0  0  0  0  0
    # 4   0  0  0  0  0  0
    # 5   1  0  1  1  0  1
    # G   0  1  0  0  1  0
    tm[5, 5, 0] = 1
    tm[5, 6, 1] = 1
    tm[5, 5, 2] = 1
    tm[5, 5, 3] = 1
    tm[5, 6, 4] = 1
    tm[5, 5, 5] = 1
    # G
    #    rt rg gt bt  c  e
    # 0   0  0  0  0  0  0
    # 1   0  0  0  0  0  0
    # 2   0  0  0  0  0  0
    # 3   0  0  0  0  0  0
    # 4   0  0  0  0  0  0
    # 5   0  0  0  0  0  0
    # G   1  1  1  1  1  1
    tm[6, 6, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 1, 1, 0]

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
    # epochs = [i for i in range(250, 7251, 250)]
    # epochs = [0, 2000, 4000, 6000, 7500]
    epochs = [7500]
    num_tests = 2
    # render_camera = False
    # env = RobotEnv(headless=True, render_camera=render_camera, option_rollout=True, episode_len=300)

    for epoch in epochs:
        results = test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=num_tests)
        save_dataset('lunchbox', metapolicy_name, task_name, epoch, results)

def test_epoch(epoch, metapolicy_class, task, metapolicy_name, num_tests=10):
    # option_load_path = os.path.join(os.environ['PKG_PATH'], 'src', 'model{}.pt'.format(epoch))
    option_load_path = os.path.join(os.environ['PKG_PATH'], 'experiments', 'ppo2', 'pyt_save', 'model{}.pt'.format(epoch))

    pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
    place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
    pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
    pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
    
    options = [pick_red_option, place_red_option, pick_green_option,
               pick_blue_option]

    print("MODEL {} | TESTING EPOCH {}, using model{}.pt".format(metapolicy_name, epoch, epoch))
    rewards, final_fsa_states, successes = run_rollout(task, options, num_tests, metapolicy_class)

    results = {'reward': rewards, 'epoch': epoch, 'success': successes, 'last_state': final_fsa_states}

    return results

###############
# Run Rollout #
###############
def run_rollout(task_spec, policies, num_episodes, metapolicy_class, cancel=False):
    _, safety_props = make_taskspec_if()
    safety_specs = make_safetyspecs_reacher()

    option_to_color = {0: 'rt', 1: 'rg', 2: 'gt', 3: 'bt'}
    goal_state = task_spec.nF - 1

    rewards = []
    final_fsa_states = []
    successes = []

    render_camera = False
    env = RobotEnv(headless=True, render_camera=render_camera, option_rollout=True, episode_len=300)


    for i in range(num_episodes):
        print('-------------------------------------------------------')
        # env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)

        cancel = (i % 2) != 0

        task_done = False
        R = 0
        obs = env.reset(cancel=cancel)
        subgoals = env.subgoals

        metapolicy = metapolicy_class(subgoals, task_spec, safety_props, safety_specs, env, policies)

        f = 0
        env_f = 0
        num_steps = 0
        prev_f = f
        success = False

        prev_option = -1
        stop_high_level = False
        terminated = False

        while not success and R > -200 and num_steps < 50:
            print("EPISODE {} STEP {} REWARD {}##################".format(i, num_steps, R))
            num_steps += 1
            done = False
            task_done = False
            if not stop_high_level:
                env.soft_reset()
                option = metapolicy.get_option(env, f)
                print("FSA: {}, option: {}, option name: {}, envf: {}".format(f, option, option_to_color[option], env_f))
                # rt = tuple(metapolicy.subgoals[0].state)
                # rg = tuple(metapolicy.subgoals[1].state)
                # gt = tuple(metapolicy.subgoals[2].state)
                # bt = tuple(metapolicy.subgoals[3].state)
                # rtik = tuple(metapolicy.subgoals[0].ik_state)[:4] + (0, 0, 0, 0)
                # rgik = tuple(metapolicy.subgoals[1].ik_state)[:4] + (0, 0, 0, 0)
                # gtik = tuple(metapolicy.subgoals[2].ik_state)[:4] + (0, 0, 0, 0)
                # btik = tuple(metapolicy.subgoals[3].ik_state)[:4] + (0, 0, 0, 0)
                # init = metapolicy.start_states[-1]
                # pdb.set_trace()
            if prev_option == option and terminated:
                stop_high_level = True
            prev_option = option
            prev_f = f
            while not done and not task_done and f == prev_f:
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

                task_done =  info['task_done']

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
            # print(done, env.get_current_propositions())
            # pdb.set_trace()

        if num_steps == 50:
            print("50 steps")
            R = -200

        rewards.append(R)
        final_fsa_states.append(f)
        successes.append(success)      
        print(f"Episode {i} return: {R} | FSA: {f}")

    env.shutdown()

    # env_can.close()
    # env_not.close()
    return rewards, final_fsa_states, successes

#######
# Run #        
#######
def run_all_tests():
    taskspec_if, _ = make_taskspec_if()
    taskspec_or, _ = make_taskspec_or()
    taskspec_sequential, _ = make_taskspec_sequential()
    taskspec_composite, _ = make_taskspec_composite()

    # metapolicies = [ContinuousMetaPolicy, GreedyContinuousMetaPolicy, FSAQLearningContinuousMetaPolicy, FlatQLearningContinuousMetaPolicy]
    # metapolicy_names = ['lof', 'greedy', 'fsa', 'flat']
    metapolicies = [ContinuousMetaPolicy]
    metapolicy_names = ['lof']
    tasks = [taskspec_composite, taskspec_sequential, taskspec_if, taskspec_or]
    task_names = ['composite', 'sequential', 'if', 'or']

    for metapolicy, metapolicy_name in zip(metapolicies, metapolicy_names):
        for task, task_name in zip(tasks, task_names):
            print("TEST {} {}".format(metapolicy_name, task_name))
            test_all_epochs(metapolicy, metapolicy_name, task, task_name)

if __name__ == '__main__':
    run_all_tests()