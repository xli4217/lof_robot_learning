import gym
import rlbench.gym

from rlbench.gym.rlbench_env import RLBenchEnv
from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig


class ReachTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

    def is_static_workspace(self) -> bool:
        return True


ReachTargetEnv = lambda render_mode: RLBenchEnv(task_class=ReachTarget, render_mode=render_mode)

if __name__ == "__main__":
    # env = ReachTargetEnv('human')
    # obs = env.reset()
    # for i in range(120):
    #     env.step(np.array([0,0,0,0,0,0,0,0]))
    #     env.render()

    ##############
    class Agent(object):

        def __init__(self, action_size):
            self.action_size = action_size

        def act(self, obs):
            arm = np.random.normal(0, 0.1, size=(self.action_size - 1,))
            gripper = [1.0]  # Always open
            return np.concatenate([arm, gripper], axis=-1)


    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    agent = Agent(env.action_size)

    training_steps = 120
    episode_length = 40
    obs = None
    for i in range(training_steps):
        if i % episode_length == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
        print(descriptions)
        action = agent.act(obs)
        #action = np.random.rand(8)
        # print(action)
        obs, reward, terminate = task.step(action)

    print('Done')
    env.shutdown()
