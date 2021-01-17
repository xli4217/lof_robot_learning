from robot_env import RobotEnv
import os
import torch
import gym
import time 


##########
# Option #
##########
class Option(object):
    def __init__(self, load_path:str):
        self.ac = torch.load(load_path)
        print(self.ac)
        print(type(self.ac))
        
    def get_action(self, state: torch.Tensor):
        return self.ac.act(state)

    def get_value(self, state: torch.Tensor):
        return self.ac.v(state)

    def is_terminated(self, state):
        return False

    def environment_info(self):
        pass


###############
# Load Option #
###############
option_load_path = os.path.join(os.environ['PKG_PATH'], 'experiments', 'test', 'pyt_save', 'model.pt')

option = Option(option_load_path)

#################
# Construct Env #
#################
env = RobotEnv(headless=False)

###############
# Run Rollout #
###############
def run_rollout(policy, env, num_episodes):
    for i in range(num_episodes):
        task_done = False
        R = 0
        obs = env.reset()
        while not task_done:
            a = policy.get_action(torch.from_numpy(obs).float())
            obs, reward, task_done, _ = env.step(a)
            R += reward
        
        time.sleep(1)
                
        print(f"Episode {i} return: {R}")

    env.shutdown()


run_rollout(option, env, 2)
