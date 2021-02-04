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


class RobotEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, headless=True, render_camera=False, option_rollout=False, episode_len=100):
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
        self.observation_space = gym.spaces.Box(low=np.array(11*[-10.]), high=np.array(11*[10.]), dtype=np.float32)
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

    def make_subgoals(self):
        # name, prop_index, subgoal_index, state
        all_info = self.all_info
        states = {
               'red_target': [0.1061546802520752, 0.6898090839385986, 0.25773075222969055, -1.8561604022979736, -0.0020071216858923435, 1.217017412185669, 0.7869886159896851],
                'red_goal': [-0.33927488327026367, 0.4343113899230957, -0.23959800601005554, -1.62320876121521, -0.0019444175995886326, 1.2168865203857422, 0.786967396736145],
                'green_target': [-0.2507748603820801, 0.617473840713501, -0.10747954249382019, -1.568331241607666, -0.0018800445832312107, 1.2168655395507812, 0.7869387865066528],
                'blue_target': [-0.24648523330688477, 0.3789811134338379, -0.1470303237438202, -1.6356382369995117, -0.0019618221558630466, 1.2168805599212646, 0.7869573831558228]
                # 'green_target': [-0.04999828338623047, 0.42098501324653625, 0.5913615226745605, -1.6058857440948486, -0.000500077847391367, 1.218733549118042, 0.7857886552810669],
                # 'blue_target': [-0.10592365264892578, 0.2675664722919464, -0.23654529452323914, -1.800999402999878, 0.002284651156514883, 1.2168364524841309, 0.7836672067642212],
                # 'red_target': [0.08351826667785645, 0.16150668263435364, -0.11060187220573425, -1.972297191619873, 0.0022682002745568752, 1.2168371677398682, 0.7836600542068481],
                # 'red_goal': [-0.08144235610961914, 0.3910515606403351, -0.6666841506958008, -1.668611764907837, 0.0023123077116906643, 1.216784954071045, 0.783650279045105]
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
            # 'green_goal': {
            #     'obj': self.green_goal,
            #     'pos': np.array(self.green_goal.get_position()),
            # },
            # 'blue_goal': {
            #     'obj': self.blue_goal,
            #     'pos': np.array(self.blue_goal.get_position()),
            # },
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
                               self.target.get_position()])

        # jp = np.array(self.agent.get_joint_positions())
        # jv = np.array(self.agent.get_joint_velocities())
        # ee_p = np.array(self.agent_ee_tip.get_position())
        # target_pos = np.array(self.target.get_position())

        # state = np.concatenate([np.cos(jp), np.sin(jp), target_pos, jv, ee_p-target_pos])

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
    
    def get_current_propositions(self, info=None, obj_name=None, threshold=0.02):
        idx_dict = {'red_target': 0, 'red_goal': 1, 'green_target': 2, 'blue_target': 3}
        if info is not None and obj_name is not None:
            props = [0]*(len(idx_dict) + 1)
            if info['grasped'] == True or info['released'] == True:
                idx = idx_dict[obj_name]
                props[idx] = 1
                return props

        state = self.agent.get_tip().get_position()
        return self.get_propositions(state, threshold)

    def get_propositions(self, state, threshold=0.02):
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
        props = [0]*(nP+1)

        for i, position in enumerate(fk_positions):
            if np.linalg.norm(state - position) < threshold:
                props[i] = 1

        if not np.any(props):
            props[-1] = 1

        return props
        
    def reset(self):
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
                # p = list(np.random.uniform(POS_MIN, POS_MAX))
                if target == 'red_target':
                    #       y  x    z
                    p = [0.95, 0.2, 0.9]
                elif target == 'green_target':
                    p = [1, -0.15, 1.0]
                elif target == 'blue_target':
                    p = [1, -0.15, 1.1]
                self.all_info[target]['obj'].set_position(p)
        
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.current_step = 0

        self.update_all_info()
        self.subgoals = self.make_subgoals()
        obs = self._get_state()
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
        
        eval = True

        if eval:
            reward = 0.2*(-0.1 + r_action)
        else:
            if dist > 0.2:
                r_dist = -dist
            else:
                r_dist = 0.01/dist
            
            reward = r_dist + r_action
            reward *= 0.2
            if dist < 0.01:
                reward += 0.5
                # done = True

        self.current_step += 1
        if self.current_step >= self.episode_len:
            #print(f"episode time: {time.time()-self.t_start}")
            done = True

        info['task_done'] = task_done

        return self._get_state(), reward, done, info

    def gripper_actuate(self, amount, velocity):
        pass
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()



if __name__ == "__main__":
        
    # class Agent(object):
        
    #     def act(self, state):
    #         del state
    #         return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    #     def learn(self, replay_buffer):
    #         del replay_buffer
    #         pass

        
    env = RobotEnv(headless=True, render_camera=True)
    env.reset()
    for _ in range(100):
        time.sleep(10)
        env.render()
    
    # agent = Agent()
    # replay_buffer = []


    # p1 = np.array([1.06900001, 0.01499999, 1.31599903])
    # p2 = np.array([1.1, 0.1, 1.2])

    # import math
    # ik1 = env.agent.solve_ik(position=p1, euler=[0, math.radians(180), 0])
    # ik2 = env.agent.solve_ik(position=p2, euler=[0, math.radians(180), 0])
    # print(f"ik1: {ik1}")
    # print(f"ik2: {ik2}")
    
    # import math
    # path = env.agent.get_path(position=[0.98,0.14,1.19], euler=[0, math.radians(180), 0])
    # print(path)
    
    # p = env.target.get_position(env.agent_ee_tip)
    # o = env.target.get_quaternion(env.agent_ee_tip)
    # print(p, o)
    # print(env.agent.get_configs_for_tip_pose(position=p, quaternion=o, relative_to=env.agent_ee_tip))
    
    # for _ in range(500):
    #     # env.gripper.grasp(env.target)
    #     while not env.gripper.actuate(amount=0.5, velocity=0.01):
    #        env.pr.step()
    #     #env.gripper.actuate(amount=0.5, velocity=0.01)
    #     action = np.array(4*[0])
    #     #action[0] = 1.
    #     reward, next_state, _, _ = env.step(action)
   
    
    # EPISODES = 2
    # for e in range(EPISODES):
    #     print('Starting episode %d' % e)
    #     state = env.reset()
    #     for i in range(EPISODE_LENGTH):
    #         #action = agent.act(state)
    #         action = np.array(4*[0])
    #         reward, next_state, _, _ = env.step(action)
    #         replay_buffer.append((state, action, reward, next_state))
    #         state = next_state
    #         agent.learn(replay_buffer)

    print('Done!')
    env.shutdown()
