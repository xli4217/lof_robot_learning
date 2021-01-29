from robot_env import RobotEnv
import os
import torch
import gym
import time 
import numpy as np

##########
# Option #
##########
class Option(object):
    def __init__(self, load_path:str, pick_or_place='pick', target='red_target'):
        self.ac = torch.load(load_path)
        print(self.ac)
        print(type(self.ac))

        self.target_name = target
        self.pick_or_place = pick_or_place
        
    def get_action(self, env_info: dict):
        state = np.concatenate([
            env_info['agent_joint_positions'],
            env_info['agent_joint_velocities'],
            env_info[self.target_name]['pos']
        ])
        
        a = self.ac.pi.mu_net(torch.from_numpy(state).float()).detach().numpy()
        if self.pick_or_place == 'pick':
            a = np.concatenate([np.array([0]), a])
        elif self.pick_or_place == 'place':
            a = np.concatenate([np.array([1]), a])
        else:
            raise ValueError('mode not supported')

        return a
        
    def get_value(self, state: torch.Tensor):
        return self.ac.v(state)

    def get_target_name(self):
        return self.target_name
        
    def is_terminated(self, state):
        return False

    def environment_info(self):
        pass

        
###############
# Load Option #
###############
option_load_path = os.path.join(os.environ['PKG_PATH'], 'src', 'model3500.pt' )
red_pick_option = Option(option_load_path, pick_or_place='pick', target='red_target')
blue_pick_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
green_pick_option = Option(option_load_path, pick_or_place='pick', target='green_target')
place_option = Option(option_load_path, pick_or_place='place', target='red_goal')

#################
# Construct Env #
#################
render_camera = False
env = RobotEnv(headless=False, render_camera=render_camera, option_rollout=True)

###############
# Run Rollout #
###############
def run_rollout(option, env, num_episodes):
    for i in range(num_episodes):
        task_done = False
        R = 0
        env.reset()
        for _ in range(200):
            a = option.get_action(env.all_info)            
            _, _, task_done, info = env.step(a, option.get_target_name())
            if render_camera:
                env.render()
            if task_done:
                break

    env.shutdown()

###############
# Run Rollout #
###############
def run_pick_and_place(rpo, bpo, gpo, place_option, env, num_episodes):
    for i in range(num_episodes):
        pick_done = False
        env.reset()
        for _ in range(1000):
            a = rpo.get_action(env.all_info)            
            _, _, pick_done, info = env.step(a, rpo.get_target_name())
            if render_camera:
                env.render()
            if pick_done:
                break
        place_done = False
        env.soft_reset()
        for _ in range(1000):
            a = place_option.get_action(env.all_info)            
            _, _, place_done, info = env.step(a, place_option.get_target_name())
            if render_camera:
                env.render()
            if place_done:
                break

        pick_done = False
        env.soft_reset()
        for _ in range(1000):
            a = bpo.get_action(env.all_info)            
            _, _, pick_done, info = env.step(a, bpo.get_target_name())
            if render_camera:
                env.render()
            if pick_done:
                break
        place_done = False
        env.soft_reset()
        for _ in range(1000):
            a = place_option.get_action(env.all_info)            
            _, _, place_done, info = env.step(a, place_option.get_target_name())
            if render_camera:
                env.render()
            if place_done:
                break

        pick_done = False
        env.soft_reset()
        for _ in range(1000):
            a = gpo.get_action(env.all_info)            
            _, _, pick_done, info = env.step(a, gpo.get_target_name())
            if render_camera:
                env.render()
            if pick_done:
                break
        place_done = False
        env.soft_reset()
        for _ in range(1000):
            a = place_option.get_action(env.all_info)            
            _, _, place_done, info = env.step(a, place_option.get_target_name())
            if render_camera:
                env.render()
            if place_done:
                break

    env.shutdown()


#### Test pick option ####
run_pick_and_place(red_pick_option, blue_pick_option, green_pick_option, place_option, env, 2)

#### Test place option ####
# run_rollout(place_option, env, 10)

