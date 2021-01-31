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
# red_pick_option = Option(option_load_path, pick_or_place='pick', target='red_target')
# blue_pick_option = Option(option_load_path, pick_or_place='pick', target='blue_target')
# green_pick_option = Option(option_load_path, pick_or_place='pick', target='green_target')
# place_option = Option(option_load_path, pick_or_place='place', target='red_goal')

pick_red_option = Option(option_load_path, pick_or_place='pick', target='red_target')
place_red_option = Option(option_load_path, pick_or_place='place', target='red_goal')
pick_green_option = Option(option_load_path, pick_or_place='pick', target='green_target')
place_green_option = Option(option_load_path, pick_or_place='place', target='green_goal')
pick_blue_option = Option(option_load_path, pick_or_place='pick', target='blue_target')


