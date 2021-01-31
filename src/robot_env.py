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
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
import gym
from spinup.algos.pytorch.ppo.ppo import ppo
import torch
import time

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

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
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
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
        self.green_goal = Shape('green_goal')
        self.blue_goal = Shape('blue_goal')
        self.red_target = Shape('red_target')
        self.green_target = Shape('green_target')
        self.blue_target = Shape('blue_target')
        
        self.update_all_info()
        
    def render(self):
        if self.render_camera:
            self._gym_cam.capture_rgb()

    def update_all_info(self):
        self.all_info = {
            'red_goal': {
                'obj': self.red_goal,
                'pos': np.array(self.red_goal.get_position()),
            },
            'green_goal': {
                'obj': self.green_goal,
                'pos': np.array(self.green_goal.get_position()),
            },
            'blue_goal': {
                'obj': self.blue_goal,
                'pos': np.array(self.blue_goal.get_position()),
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
                               self.target.get_position()])

        # jp = np.array(self.agent.get_joint_positions())
        # jv = np.array(self.agent.get_joint_velocities())
        # ee_p = np.array(self.agent_ee_tip.get_position())
        # target_pos = np.array(self.target.get_position())

        # state = np.concatenate([np.cos(jp), np.sin(jp), target_pos, jv, ee_p-target_pos])

        return state
        
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
                p = list(np.random.uniform(POS_MIN, POS_MAX))
                self.all_info[target]['obj'].set_position(p)
        
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.current_step = 0

        self.update_all_info()
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
                    task_done = True
                    print("Graspped")
                else:
                    info['grasped'] = False
            elif action[0] == 1 and np.linalg.norm(np.array(self.agent_ee_tip.get_position()) - np.array(self.all_info[obj_name]['pos'])) < 0.05: # open
                self.gripper.release()
                while not self.gripper.actuate(amount=1., velocity=0.01):
                    self.pr.step()
                info['released'] = True
                task_done = True
                print("Released")
        r_action = - np.linalg.norm(action)
        
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
        
    class Agent(object):
        
        def act(self, state):
            del state
            return list(np.random.uniform(-1.0, 1.0, size=(7,)))

        def learn(self, replay_buffer):
            del replay_buffer
            pass

        
    env = RobotEnv(headless=True, render_camera=False)
    agent = Agent()
    replay_buffer = []

    #env.agent.set_ik_group_properties(resolution_method='damped_least_squares', dls_damping=0.1)
    #env.agent.set_ik_element_properties(constraint_alpha_beta=False, constraint_gamma=False)

    env.agent.set_joint_positions(env.agent.get_joint_positions())
    print(env.agent.get_joint_positions())
    p1 = env.target.get_position()
    q1 = env.target.get_quaternion()
    o1 = env.target.get_orientation()
    #env.agent.get_configs_for_tip_pose(position=p1, quaternion=o1, relative_to=env.agent)

    import math
    path = env.agent.get_path(position=[0.98,0.14,1.19], euler=[0, math.radians(180), 0])
    print(path)
    
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
