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
pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
place_green_option = Option(option_load_path, pick_or_place='place', target='green_goal')

#################
# Construct Env #
#################
render_camera = False
env = RobotEnv(headless=False, render_camera=render_camera, option_rollout=True, episode_len=200)

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
            print(f"using option {option_name}")
            done = False
            while not done and not task_done:
                a = option.get_action(env.all_info)            
                _, _, done, info = env.step(a, option.get_target_name())
                if render_camera:
                    env.render()
                task_done = info['task_done']
                if done or task_done:
                    env.current_step = 0
                    done = False
                    task_done = False
                    break
    env.shutdown()


#### Test pick and place option ####
options_dict = {
    'pick_red1': pick_red_option,
    'place_red2': place_red_option,
    'pick_green3': pick_green_option,
    'place_red4': place_red_option
}
run_rollout(options_dict, env, 10)

#### Test place option ####
#run_rollout(place_option, env, 10)

