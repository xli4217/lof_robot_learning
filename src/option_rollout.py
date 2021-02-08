from robot_env import RobotEnv
import os
import torch
import gym
import time 
import numpy as np
from option import Option
##########
# Option #
##########
# class Option(object):
#     def __init__(self, load_path:str, pick_or_place='pick', target='red_target'):
#         self.ac = torch.load(load_path)
#         print(self.ac)
#         print(type(self.ac))

#         self.target_name = target
#         self.pick_or_place = pick_or_place
        
#     def get_action(self, env_info: dict):
#         state = np.concatenate([
#             env_info['agent_joint_positions'],
#             env_info['agent_joint_velocities'],
#             env_info[self.target_name]['pos']
#         ])
        
#         a = self.ac.pi.mu_net(torch.from_numpy(state).float()).detach().numpy()
#         if self.pick_or_place == 'pick':
#             a = np.concatenate([np.array([0]), a])
#         elif self.pick_or_place == 'place':
#             a = np.concatenate([np.array([1]), a])
#         else:
#             raise ValueError('mode not supported')

#         return a
        
#     # state is 11 dimensional
#     # [j0 j1 j2 j3 vj0 vj1 vj2 vj3 x y z]
#     def get_value(self, state: torch.Tensor):
#         return self.ac.v(state)

#     def get_target_name(self):
#         return self.target_name
        
#     def is_terminated(self, state):
#         return False

#     def environment_info(self):
#         pass

        
###############
# Load Option #
###############
option_load_path = os.path.join(os.environ['PKG_PATH'], 'experiments', 'ppo2', 'pyt_save', 'model6000.pt' )

pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')

#################
# Construct Env #
#################
render_camera = False
env = RobotEnv(headless=False, render_camera=render_camera, option_rollout=True, episode_len=300)

###############
# Run Rollout #
###############
def run_rollout(options, env, num_episodes):
    for i in range(num_episodes):
        done = False
        task_done = False
        R = 0
        env.reset()
        for option_name, option in options.items():
            env.soft_reset()
            print(f"using option {option_name}")
            done = False
            while not done and not task_done:
                # a is 5 dimensional. The first dimension is open/close gripper
                a = option.get_action(env.all_info)            
                _, _, done, info = env.step(a, option.get_target_name())
                if render_camera:
                    env.render()
                task_done = info['task_done']
                if done or task_done:
                    print(env.agent.get_joint_positions())
                    env.current_step = 0
                    done = False
                    task_done = False
                    break
    env.shutdown()

#### Test pick option ####
# run_pick_and_place(red_pick_option, blue_pick_option, green_pick_option, place_option, env, 2)

#### Test pick and place option ####
options_dict = {
    'pick_green': pick_green_option,
    'place_red1': place_red_option,
    'pick_blue': pick_blue_option,
    'place_red2': place_red_option,
    'pick_red': pick_red_option,
    'place_red3': place_red_option
}
run_rollout(options_dict, env, 1)

#### Test place option ####
# run_rollout(place_option, env, 10)

