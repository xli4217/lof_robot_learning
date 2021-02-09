"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
import os

# cwd = os.getcwd()
# coppeliasim_root = cwd  + '/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04'
# os.environ['LD_LIBRARY_PATH'] = coppeliasim_root + ':' + os.environ['LD_LIBRARY_PATH']
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = coppeliasim_root

# spinningup_root = cwd + '/libs/spinningup'
# rlbench_root = cwd + '/libs/RLBench'
# os.environ['PYTHONPATH'] = spinningup_root + ":" + os.environ['PYTHONPATH']
# os.environ['PYTHONPATH'] = rlbench_root + ":" + os.environ['PYTHONPATH']
# os.environ['PKG_PATH'] = cwd

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
import gym
from spinup.algos.pytorch.ppo.ppo import ppo
import torch
import time
import pdb

from qrm.src.reward_machines.reward_machine import RewardMachine

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

from option import Subgoal

SCENE_FILE = join(dirname(abspath(__file__)), 'ttt',
                  'scene_reinforcement_learning_env.ttt')
#POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
sample_region_dim = np.array([0.3, 0.7, 0.4])
sample_region_pos = np.array([0.892,0, 0.961])
POS_MIN, POS_MAX = list(sample_region_pos - sample_region_dim/2), list(sample_region_pos + sample_region_dim/2) 


class RMRobotEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, nF=0, task_name='', can_chance=0.5, eval=False, headless=True, render_camera=False, option_rollout=False, episode_len=100):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Panda()
        
        # so that you don't have to do forward kinematics over and over again,
        # store a dictionary
        self.fk_dict = {}

        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        # self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        #self.gripper_sensor = ProximitySensor('Panda_gripper_grasp_sensor')
        self.option_rollout = option_rollout
        self.episode_len = episode_len
        
        self.gripper = PandaGripper()
        self.gripper_dummy = Dummy('Panda_tip')
        
        self.agent_config_tree = self.agent.get_configuration_tree()

        self._render_mode = 'human'
        if self.option_rollout:
            action_dim = 5 # gripper state and first 4 joint velocities
            POS_MIN[2] += 0.3
            POS_MAX[2] += 0.2
        else:
            action_dim = 4 # first 4 joint velocities

        action_high = np.array(action_dim * [1.])
        action_low = np.array(action_dim * [-1.])

        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array(21*[-10.]), high=np.array(21*[10.]), dtype=np.float32)
        self.current_step = 0

        self.render_camera = render_camera
        if self.render_camera:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)

        
        self.red_goal = Shape('red_goal')
        # self.green_goal = Shape('green_goal')
        # self.blue_goal = Shape('blue_goal')
        self.red_target = Shape('red_target')
        self.green_target = Shape('green_target')
        self.blue_target = Shape('blue_target')

        self.target = self.red_target
        
        self.update_all_info()

        self.subgoals = None
        self.can_state = 1
        self.cancel = np.random.uniform() < can_chance

        self.eval = eval
        self.nF = nF
        self.f = 0
        # initialize RM
        self.task_name = task_name
        rm_file = os.path.join(os.environ['PKG_PATH'], 'libs', 'qrm', 'rm', 'tasks', task_name + '.txt')

        self.rm = RewardMachine(rm_file, use_rs=False, gamma=0.9)

    def make_subgoals(self):
        # name, prop_index, subgoal_index, state
        all_info = self.all_info
        states = {
               'red_target': [-0.024184703826904297, 0.5780532360076904, 0.49106720089912415, -1.7368626594543457, -0.0010691829957067966, 1.2176384925842285, 0.7865573167800903],
                'red_goal': [-0.6136815547943115, 0.4038100242614746, -0.12854143977165222, -1.657426357269287, -0.0010327049531042576, 1.2176005840301514, 0.7865227460861206],
                'green_target': [-0.34819626808166504, 0.8475337028503418, 0.023823529481887817, -1.4239075183868408, -0.001070136670023203, 1.2176568508148193, 0.7865363359451294],
                'blue_target': [-0.47508692741394043, 0.28998875617980957, 0.03704550862312317, -1.6011130809783936, -0.0010186382569372654, 1.2176315784454346, 0.7865163087844849]
            }
        red_target = Subgoal('red_target', 0, 0, all_info['red_target']['pos'], states['red_target'])
        red_goal = Subgoal('red_goal', 1, 1, all_info['red_goal']['pos'], states['red_goal'])
        green_target = Subgoal('green_target', 2, 2, all_info['green_target']['pos'], states['green_target'])
        blue_target = Subgoal('blue_target', 3, 3, all_info['blue_target']['pos'], states['blue_target'])
        
        return [red_target, red_goal, green_target, blue_target]

        
    def render(self):
        if self.render_camera:
            self._gym_cam.capture_rgb()

    def update_all_info(self):
        self.all_info = {
            'red_goal': {
                'obj': self.red_goal,
                'pos': np.array(self.red_goal.get_position()),
            },
            'red_target': {
                'obj': self.red_target,
                'pos': np.array(self.red_target.get_position()),
            },
            'green_target': {
                'obj': self.green_target,
                'pos': np.array(self.green_target.get_position()),
            },
            'blue_target': {
                'obj': self.blue_target,
                'pos': np.array(self.blue_target.get_position()),
            },
            'agent_joint_positions': self.agent.get_joint_positions()[:4],
            'agent_joint_velocities': self.agent.get_joint_velocities()[:4]
        }

            
    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position        
        return np.concatenate([self.agent.get_joint_positions()[:4],
                               self.agent.get_joint_velocities()[:4],
                               self.red_target.get_position(),
                               self.red_goal.get_position(),
                               self.green_target.get_position(),
                               self.blue_target.get_position()])
        return state

    # this is for when you're running multiple options in a row
    # and you don't want to reset the environment, just reset
    # a few parameters before the option runs
    def soft_reset(self):
        self.current_step = 0

    # convert [j0, j1, j2, j3, vj0, vj1, vj2, vj3]
    # to [x, y, z]
    def fk_state(self, state):
        initial_state = self.agent.get_joint_positions()

        # if tuple(initial_state) in self.fk_dict:
            # return np.array(self.fk_dict[tuple(initial_state)])
        # else:
        new_state = self.agent.get_joint_positions()
        new_state[:4] = state[:4]
        if tuple(new_state[:4]) in self.fk_dict:
            return np.array(self.fk_dict[tuple(new_state[:4])])
        else:
            # print('not in fk_dict: ', tuple(new_state[:4]))
            self.agent.set_joint_positions(new_state)
            fk_state = self.agent.get_tip().get_position()
            self.fk_dict[tuple(new_state[:4])] = tuple(fk_state)
            self.agent.set_joint_positions(initial_state)
            # print(new_state, fk_state)
            return fk_state
    
    def get_current_propositions(self, info=None, obj_name=None, threshold=0.02, f=None):
        idx_dict = {'red_target': 0, 'red_goal': 1, 'green_target': 2, 'blue_target': 3}
        if info is not None and obj_name is not None:
            props = [0]*(len(idx_dict) + 2)
            if info['grasped'] == True or info['released'] == True:
                idx = idx_dict[obj_name]
                props[idx] = 1
                return props

        state = self.agent.get_tip().get_position()
        return self.get_propositions(state, threshold, f)

    def get_propositions(self, state, threshold=0.02, f=None):
        # state = [x, y, z]
        state = np.array(state)
        if len(state) > 3:
            state = self.fk_state(state)

        red_target = self.subgoals[0].ik_state
        red_goal = self.subgoals[1].ik_state
        green_target = self.subgoals[2].ik_state
        blue_target = self.subgoals[3].ik_state

        positions = [red_target, red_goal, green_target, blue_target]
        fk_positions = []
        for p in positions:
            fk_p = self.fk_state(p)
            fk_positions.append(fk_p)
        nP = len(positions)
        props = [0]*(nP+2)

        for i, position in enumerate(fk_positions):
            if np.linalg.norm(state - position) < threshold:
                props[i] = 1

        # can prop
        if f == self.can_state and not np.any(props) and self.cancel:
            print("CANCELLED!!!")
            props[-2] = 1
        # empty prop
        elif not np.any(props):
            props[-1] = 1

        return props
        
    def reset(self, cancel=False):
        print("RESETINGGGGGGGGGGGGGGGGGGGGGGGGGG")
        self.cancel = cancel
        # Get a random position within a cuboid and set the target position
        self.t_start = time.time()
        self.pr.set_configuration_tree(self.agent_config_tree)

        # reset gripper
        self.gripper.release()
        while not self.gripper.actuate(amount=1., velocity=0.01):
            self.pr.step()


        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)

        if self.option_rollout:
            for target in ['red_target', 'green_target', 'blue_target']:
                if not self.eval:
                    p = list(np.random.uniform(POS_MIN, POS_MAX))
                else:
                    if target == 'red_target':
                        #       y  x    z
                        p = [1, 0.25, 1]
                    elif target == 'green_target':
                        p = [1, -0.15, 0.9]
                    elif target == 'blue_target':
                        p = [1, -0.15, 1.1]
                self.all_info[target]['obj'].set_position(p)
        
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.current_step = 0

        self.update_all_info()
        self.subgoals = self.make_subgoals()
        obs = self._get_state()
        if not self.eval:
            self.f = np.random.randint(self.nF - 1)
            print("Init FSA state: ", self.f)
        else:
            self.f = 0
        print("TIP POSITION: ", self.agent.get_tip().get_position(), self.all_info['agent_joint_positions'])
        obs = np.concatenate([np.array([self.f]), obs])
        return obs

    def step(self, action, obj_name=None):
        done = False
        task_done = False
        info = {
            'grasped': False,
            'released': False
        }

        if not self.option_rollout:
            joint_action = np.concatenate([action, np.zeros(3)])
        else:
            joint_action = np.concatenate([action[1:], np.zeros(3)])
            
        for _ in range(1):
            self.agent.set_joint_target_velocities(joint_action)  # Execute action on arm
            self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()

        self.update_all_info()
        
        # Reward is negative distance to target
        dist = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

        # gripper control
        if self.option_rollout:
            if action[0] == 0: # close
                #print(obj_name)
                #print(self.gripper.grasp(self.all_info[obj_name]['obj']))
                if self.gripper.grasp(self.all_info[obj_name]['obj']):
                    while not self.gripper.actuate(amount=0.5, velocity=0.01):
                        self.pr.step()
                    info['grasped'] = True
                    done = True
                    print("Grasped")
                else:
                    info['grasped'] = False
            elif action[0] == 1 and np.linalg.norm(np.array(self.agent_ee_tip.get_position()) - np.array(self.all_info[obj_name]['pos'])) < 0.08: # open
                self.gripper.release()
                while not self.gripper.actuate(amount=1., velocity=0.01):
                    self.pr.step()
                info['released'] = True
                task_done = True
                print("Released")
        r_action = - np.linalg.norm(action)
        
        ########## RM Stuff
        tip = self.agent.get_tip().get_position()
        dist_red = np.linalg.norm(tip - self.all_info['red_target']['pos'])
        dist_rg = np.linalg.norm(tip - self.all_info['red_goal']['pos'])
        dist_green = np.linalg.norm(tip - self.all_info['green_target']['pos'])
        dist_blue = np.linalg.norm(tip - self.all_info['blue_target']['pos'])

        true_props = []
        if dist_red < 0.02:
            true_props = ['r']
        elif dist_rg < 0.02:
            true_props = ['y']
        elif dist_green < 0.02:
            true_props = ['g']
        elif dist_blue < 0.02:
            true_props = ['b']
        elif self.f == self.can_state and self.cancel:
            true_props = ['c']
        if true_props != []:
            print("PROP: ", true_props)
            
        next_f = self.rm.get_next_state(self.f, true_props)
        # print(next_f)

        reward = 0
        if next_f != self.f:
            reward += 1
            print("FSA STATE CHANGED FROM {} TO {}".format(self.f, next_f))

        if next_f != self.nF - 1:
            reward += -0.1 # - np.square(action).sum() - 0.1
        else:
            print("goal state")
            reward += 1
            done = True
            info['task_done'] = True

        self.f = next_f

        ###############

        if self.eval:
            reward = 0.2*(-0.1 + r_action)
        # else:
        #     if dist > 0.2:
        #         r_dist = -2*dist
        #     else:
        #         r_dist = -dist
        #     reward = r_dist + r_action
        #     reward *= 0.2
        #     if dist < 0.02:
        #         reward = 0 

        self.current_step += 1
        if self.current_step >= self.episode_len:
            #print(f"episode time: {time.time()-self.t_start}")
            done = True

        # info['task_done'] = task_done

        obs = np.concatenate([np.array([self.f]), self._get_state()])


        return obs, reward, done, info

    def gripper_actuate(self, amount, velocity):
        pass
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()



if __name__ == "__main__":
        
    env = RMRobotEnv('OR', headless=True, render_camera=True)
    env.reset()
    for _ in range(100):
        time.sleep(10)
        env.render()

    print('Done!')
    env.shutdown()
