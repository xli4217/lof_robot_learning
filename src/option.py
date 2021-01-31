import torch
import os
import numpy as np

# changed state_index to just state
class Subgoal(object):

    def __init__(self, name, prop_index, subgoal_index, state):
        self.name = name
        # index in the overall list of props
        self.prop_index = prop_index
        # index in the list of subgoals (theoretically
        # the same as the prop_index)
        self.subgoal_index = subgoal_index
        self.state = state

# no modifications from discrete case
class TaskSpec(object):

    def __init__(self, task_spec, tm, task_state_costs):
        self.spec = task_spec
        self.tm = tm
        self.nF = tm.shape[0]
        self.task_state_costs = task_state_costs


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

class MetaPolicyBase(object):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3):
        self.task_spec = task_spec
        self.tm = task_spec.tm
        # instances of Subgoal
        self.subgoals = subgoals
        # assume it's the index of each safety prop in the list of props
        self.safety_props = safety_props
        # a list of safety specs, one for each subgoal
        self.safety_specs = safety_specs

        # number of options
        self.nO = len(self.subgoals)
        # number of safety props
        self.nS = len(self.safety_props)

        self.options = None
        self.reward = None
        self.poss = None

        # training parameters
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def init_poss(self, subgoals):
        raise NotImplementedError

# LOF class for metapolicies on continuous spaces
class ContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, options,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=5, experiment_num=0, num_hq_iter=1000):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.experiment_num = experiment_num

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

        # basically, a list of option starting states
        # aka the initial state and the goal states
        self.start_states = [tuple(env.all_info['ee_p'])]
        for subgoal in self.subgoals:
            self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.options = options
        # self.rewards[(start_state[x, y], subgoal_index)] = reward
        # start states consist of initial state and goal states
        # the reward function is a function over start states and the goal indices
        # I chose to use goal indices rather than goal states because when the agent reaches a goal,
        # it will not arrive at the exact goal state. So when I will need to replan after arriving at a goal,
        # which will involve adding a new start state that is also technically a goal state.
        # but I don't want to mess with the goal states, just the start states.
        self.option_rewards = self.make_option_reward_function(env)
        self.poss = self.make_poss(env.all_info['ee_p'])

        self.num_hq_iter = num_hq_iter # number of high q iterations
        self.Q = self.make_metapolicy(env, task_spec, num_iter=num_hq_iter)

    # the 'transition function' of the options, a list of dictionaries of the form
    # self.poss = [option][start state : end state]
    def make_poss(self, start_state):
        posses = []
        for subgoal in self.subgoals:
            poss = {}
            poss[tuple(start_state)] = tuple(subgoal.state)
            poss[tuple(subgoal.state)] = tuple(subgoal.state)
            posses.append(poss)
        return posses

    # do inverse kinematics to get the joint angles
    # of the 2-link reacher when the end effector
    # is at a given goal
    # note that the link lengths are assumed to be
    # 0.1 and 0.1
    # using equations from https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/
    def get_thetas_for_state(self, state):
        x = state[0]
        y = state[1]
        a1 = 0.1
        a2 = 0.1
        # there are two possible solutions to the IK problem
        # solution 1
        q2_1 = np.arccos((x**2 + y**2 - a1**2 - a2**2)/(2*a1*a2))
        q1_1 = np.arctan2(y, x) - np.arctan2(a2*np.sin(q2_1), a1 + a2*np.cos(q2_1))
        # solution 2
        q2_2 = -q2_1
        q1_2 = np.arctan2(y, x) + np.arctan2(a2*np.sin(q2_2), a1 + a2*np.cos(q2_2))

        return [[q1_1, q2_1], [q1_2, q2_2]]

    # get the reward for going from a start state to a goal state
    def get_reward_for_state(self, env, start_state, goal_state):
        joint_velocity = [0, 0]

        jp_1, jp_2 = self.get_thetas_for_state(goal_state)

        state1 = np.concatenate([
            np.cos(jp_1),
            np.sin(jp_1),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        state2 = np.concatenate([
            np.cos(jp_2),
            np.sin(jp_2),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        value1 = self.option.get_value(torch.Tensor(state1).float())
        value2 = self.option.get_value(torch.Tensor(state2).float())

        return max(value1, value2)

    def make_option_reward_function(self, env):
        rewards = []
        for subgoal in self.subgoals:        
            reward = {}
            for start_state in self.start_states:
                goal_state = subgoal.state
                reward[tuple(start_state)] = self.get_reward_for_state(env, start_state, goal_state).item()
            rewards.append(reward)
        return rewards

    def is_terminated(self, env, state, option):
        props = env.get_propositions(state)
        if props[option] == 1 or props[option+5] == 1:
            return True
        else:
            return False
        # return self.option.is_terminated(state)

    def get_fsa_state(self, env, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_current_propositions(threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def init_value_function(self, nF):
        V = {}
        for i in range(nF):
            for s in self.start_states:
                V[(i, tuple(s))] = 0.
                for subgoal in self.subgoals:
                    ns = tuple(subgoal.state)
                    V[(i, ns)] = 0.
                # for i, poss in enumerate(self.poss):
                #     ns = self.subgoals[i].state
                #     # ns = poss[tuple(s)]
                #     V[(i, tuple(ns))] = 0.
        return V

    def make_reward_function(self, task_spec):
        return task_spec.task_state_costs

    def make_metapolicy(self, env, task_spec, num_iter=1000):
        # TM: f x f x p
        # P: p x s xxxxxxxxxxxxxxxxxxx

        # T: a x s x s (a list over a) xxxxxxxxxxxxxxxx

        # self.option_rewards: [option][start state: reward]
        # self.poss : [option]{start state : end state}

        # self.nF
        # self.nO

        # R: [f] = reward
        R = self.make_reward_function(task_spec)
        # V: {(f x s)} = value
        V = self.init_value_function(self.nF)

        # Q[(option, state, option)]
        Q = {}
        for k in range(num_iter):
            for f in range(self.nF):
                for s in self.start_states:
                    s = tuple(s)
                    best_o = 0
                    best_o_value = -np.inf
                    for o, poss in enumerate(self.poss):
                        # if s not in poss.keys():
                            # print("{} was not in poss.keys()".format(s))
                            # poss[s] = s
                        if s not in self.option_rewards[o].keys():
                            print("{} was not in rewards[{}].keys()".format(s, o))
                            self.option_rewards[o][s] = -1.
                        
                        ns = tuple(self.subgoals[o].state)
                        props = env.get_propositions(ns, threshold=0.02)
                        p = np.where(np.array(props) == 1)
                        nf = np.argmax(self.tm[f, :, p])
                        # ns = tuple(poss[s])
                        Q[(f, s, o)] = float(R[f])*float(self.option_rewards[o][s] - 1) + float(V[(nf, ns)])
                        # if f == 3 and o == 2:
                            # print('f')
                        if Q[(f, s, o)] > best_o_value:
                            best_o_value = Q[(f, s, o)]
                            best_o = o
                    V[(f, s)] = Q[(f, s, best_o)]
            
            for s in self.start_states:
                s = tuple(s)
                for f in range(self.nF):
                    props = env.get_propositions(s, threshold=0.02)
                    p = np.where(np.array(props) == 1)
                    nf = np.argmax(self.tm[f, :, p])
                    V[(f, s)] = V[(nf, s)]

        return Q

    def listQ(self, Q):
        for key, value in Q.items():
            print(key, value)

    def listV(self, V):
        for key, value in V.items():
            print(key, value)

    def get_closest_state(self, state):
        closest_state = None
        closest_dist = np.inf
        for s in self.start_states:
            dist = np.linalg.norm(np.array(state) - np.array(s))
            if dist < closest_dist and dist < 0.02:
                closest_dist = dist
                closest_state = s
        if closest_state is None:
            closest_state = state
        return tuple(closest_state)

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = self.get_closest_state(tuple(env.all_info['ee_p']))
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            self.poss = self.make_poss(state)
            self.Q = self.make_metapolicy(env, self.task_spec, num_iter=self.num_hq_iter)
            # print('hi')

        best_option = 0
        best_option_value = -np.inf
        for o in range(len(self.subgoals)):
            if self.Q[(f, state, o)] > best_option_value:
                best_option = o
                best_option_value = self.Q[(f, state, o)]

        return best_option

# LOF class for metapolicies on continuous spaces
class FSAQLearningContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, option,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=5, experiment_num=0, num_hq_iter=1000):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.experiment_num = experiment_num

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

        # basically, a list of option starting states
        # aka the initial state and the goal states
        self.start_states = [tuple(env.all_info['ee_p'])]
        for subgoal in self.subgoals:
            self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.option = option
        # self.option_rewards[option][(start_state(x, y)] = reward
        self.option_rewards = self.make_option_reward_function(env)

        self.Q = self.high_level_q_learning(env, tuple(env.all_info['ee_p']), num_hq_iter=num_hq_iter)

    # do inverse kinematics to get the joint angles
    # of the 2-link reacher when the end effector
    # is at a given goal
    # note that the link lengths are assumed to be
    # 0.1 and 0.1
    # using equations from https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/
    def get_thetas_for_state(self, state):
        x = state[0]
        y = state[1]
        a1 = 0.1
        a2 = 0.1
        # there are two possible solutions to the IK problem
        # solution 1
        q2_1 = np.arccos((x**2 + y**2 - a1**2 - a2**2)/(2*a1*a2))
        q1_1 = np.arctan2(y, x) - np.arctan2(a2*np.sin(q2_1), a1 + a2*np.cos(q2_1))
        # solution 2
        q2_2 = -q2_1
        q1_2 = np.arctan2(y, x) + np.arctan2(a2*np.sin(q2_2), a1 + a2*np.cos(q2_2))

        return [[q1_1, q2_1], [q1_2, q2_2]]

    # get the reward for going from a start state to a goal state
    def get_reward_for_state(self, env, start_state, goal_state):
        joint_velocity = [0, 0]

        jp_1, jp_2 = self.get_thetas_for_state(goal_state)

        state1 = np.concatenate([
            np.cos(jp_1),
            np.sin(jp_1),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        state2 = np.concatenate([
            np.cos(jp_2),
            np.sin(jp_2),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        value1 = self.option.get_value(torch.Tensor(state1).float())
        value2 = self.option.get_value(torch.Tensor(state2).float())

        return max(value1, value2)

    def make_option_reward_function(self, env):
        rewards = []
        for subgoal in self.subgoals:        
            reward = {}
            for start_state in self.start_states:
                goal_state = subgoal.state
                reward[tuple(start_state)] = self.get_reward_for_state(env, start_state, goal_state).item()
            rewards.append(reward)
        return rewards

    def listQ(self, Q):
        for key, value in Q.items():
            print(key, value)

    def listV(self, V):
        for key, value in V.items():
            print(key, value)

    def is_terminated(self, env, state, option):
        props = env.get_propositions(state)
        if props[option] == 1 or props[option+5] == 1:
            return True
        else:
            return False

    def get_fsa_state(self, env, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_current_propositions(threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def get_fsa_state_helper(self, env, state, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_propositions(state, threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def make_reward_function(self, task_spec):
        return task_spec.task_state_costs

    def get_best_option(self, Q, f, state):
        state = tuple(state)
        best_option = 0
        best_option_value = -np.inf
        for o in range(len(self.subgoals)):
            if Q[(f, state, o)] > best_option_value:
                best_option = o
                best_option_value = Q[(f, state, o)]
        return best_option, best_option_value

    def high_level_q_learning(self, env, start_state, start_f=0, alpha=0.5, task_spec=None, num_hq_iter=1000):
        if task_spec is None:
            task_spec = self.task_spec

        nF = task_spec.nF

        Q, V = self.init_Q_and_value_function(task_spec)

        num_episodes = num_hq_iter # 2000
        episode_length = 15
        gamma = 1.
        epsilon = 0.4

        start_state = tuple(start_state)
        
        goal_state = task_spec.nF - 1
        for i in range(num_episodes):
            f = start_f # i % (nF - 1) # cycle through the FSA states except the goal state
            num_steps = 0
            current_state = start_state
            while num_steps < episode_length:
                if np.random.uniform() < epsilon:
                    option_index = np.random.choice(range(len(self.subgoals)))
                else:
                    option_index, _ = self.get_best_option(Q, f, current_state)

                next_state = tuple(self.subgoals[option_index].state)
                next_f = self.get_fsa_state_helper(env, next_state, f, task_spec.tm)

                reward = task_spec.task_state_costs[f] * (self.option_rewards[option_index][current_state] - 1)

                q_update = reward + gamma * V[(next_f, next_state)] - Q[(f, current_state, option_index)]
                Q[f, current_state, option_index] += alpha * q_update

                for f in range(task_spec.nF):
                    for s in self.start_states:
                        s = tuple(s)
                        _, best_value = self.get_best_option(Q, f, s)
                        V[(f, s)] = best_value

                f = next_f
                current_state = next_state
                num_steps += 1

        return Q

    def init_Q_and_value_function(self, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec
        V = {}
        Q = {}
        for f in range(task_spec.nF):
            for s in self.start_states:
                s = tuple(s)
                V[(f, s)] = 0.

                for o, subgoal in enumerate(self.subgoals):
                    Q[(f, s, o)] = 0.
        
        return Q, V

    def get_closest_state(self, state):
        closest_state = None
        closest_dist = np.inf
        for s in self.start_states:
            dist = np.linalg.norm(np.array(state) - np.array(s))
            if dist < closest_dist and dist < 0.02:
                closest_dist = dist
                closest_state = s
        if closest_state is None:
            closest_state = state
        return tuple(closest_state)

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = self.get_closest_state(tuple(env.all_info['ee_p']))
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            self.Q = self.high_level_q_learning(env, state, start_f=f)
            # print('hi')

        best_option = 0
        best_option_value = -np.inf
        for o in range(len(self.subgoals)):
            if self.Q[(f, state, o)] > best_option_value:
                best_option = o
                best_option_value = self.Q[(f, state, o)]

        return best_option

# LOF class for metapolicies on continuous spaces
class FlatQLearningContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, option,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=5, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.experiment_num = experiment_num

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

        # basically, a list of option starting states
        # aka the initial state and the goal states
        self.start_states = [tuple(env.all_info['ee_p'])]
        for subgoal in self.subgoals:
            self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.option = option
        self.option_rewards = self.make_option_reward_function(env)

        self.Q = self.high_level_q_learning(env, tuple(env.all_info['ee_p']))

    def is_terminated(self, env, state, option):
        props = env.get_propositions(state)
        if props[option] == 1 or props[option+5] == 1:
            return True
        else:
            return False

    def get_best_option(self, Q, state):
        state = tuple(state)
        best_option = 0
        best_option_value = -np.inf
        for o in range(len(self.subgoals)):
            if Q[(state, o)] > best_option_value:
                best_option = o
                best_option_value = Q[(state, o)]
        return best_option, best_option_value

    def high_level_q_learning(self, env, start_state, alpha=0.5):        
        if tuple(start_state) not in self.start_states:
            self.start_states.append(tuple(start_state))
            self.option_rewards = self.make_option_reward_function(env)
        
        # V[tuple(state)] = value
        # Q[(tuple(state), option)] = Q value
        Q, V = self.init_Q_and_value_function()

        num_episodes = 200
        episode_length = 10
        gamma = 1
        epsilon = 0.3

        # need to keep track of the true FSA state so that a goal reward can be made
        # and so you can stop learning when the goal state is reached
        f = 0
        goal_state = self.task_spec.nF - 1
        for i in range(num_episodes):
            num_steps = 0
            current_state = start_state
            while num_steps < episode_length and f != goal_state:
                if np.random.uniform() < epsilon:
                    option_index = np.random.choice(range(len(self.subgoals)))
                else:
                    option_index, _ = self.get_best_option(Q, current_state)

                next_state = tuple(self.subgoals[option_index].state)

                f = self.get_fsa_state_helper(env, next_state, f)

                if f == goal_state:
                    reward = 0
                else:
                    reward = self.option_rewards[option_index][current_state]
                    # to compensate for the fact that option reward is 0 at the goal
                    # but should be -1 for the overall policy
                    if reward == 0:
                        reward = -1

                q_update = reward + gamma * V[next_state] - Q[(current_state, option_index)]
                Q[(current_state, option_index)] += alpha * q_update
                for s in self.start_states:
                    s = tuple(s)
                    _, best_value = self.get_best_option(Q, s)
                    V[s] = best_value

                current_state = next_state

                num_steps += 1

        return Q



    # do inverse kinematics to get the joint angles
    # of the 2-link reacher when the end effector
    # is at a given goal
    # note that the link lengths are assumed to be
    # 0.1 and 0.1
    # using equations from https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/
    def get_thetas_for_state(self, state):
        x = state[0]
        y = state[1]
        a1 = 0.1
        a2 = 0.1
        # there are two possible solutions to the IK problem
        # solution 1
        q2_1 = np.arccos((x**2 + y**2 - a1**2 - a2**2)/(2*a1*a2))
        q1_1 = np.arctan2(y, x) - np.arctan2(a2*np.sin(q2_1), a1 + a2*np.cos(q2_1))
        # solution 2
        q2_2 = -q2_1
        q1_2 = np.arctan2(y, x) + np.arctan2(a2*np.sin(q2_2), a1 + a2*np.cos(q2_2))

        return [[q1_1, q2_1], [q1_2, q2_2]]

    # get the reward for going from a start state to a goal state
    def get_reward_for_state(self, env, start_state, goal_state):
        joint_velocity = [0, 0]

        jp_1, jp_2 = self.get_thetas_for_state(goal_state)

        state1 = np.concatenate([
            np.cos(jp_1),
            np.sin(jp_1),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        state2 = np.concatenate([
            np.cos(jp_2),
            np.sin(jp_2),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        value1 = self.option.get_value(torch.Tensor(state1).float())
        value2 = self.option.get_value(torch.Tensor(state2).float())

        return max(value1, value2)

    def make_option_reward_function(self, env):
        rewards = []
        for subgoal in self.subgoals:        
            reward = {}
            for start_state in self.start_states:
                goal_state = subgoal.state
                reward[tuple(start_state)] = self.get_reward_for_state(env, start_state, goal_state).item()
            rewards.append(reward)
        return rewards

    def get_fsa_state(self, env, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_current_propositions(threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def get_fsa_state_helper(self, env, state, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_propositions(state, threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def init_Q_and_value_function(self):
        V = {}
        for s in self.start_states:
            V[tuple(s)] = 0.

        Q = {}
        for s in self.start_states:
            for i, subgoal in enumerate(self.subgoals):
                Q[(tuple(s), i)] = 0.
        
        return Q, V

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = tuple(env.all_info['ee_p'])
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            self.Q = self.high_level_q_learning(env, state)

        best_option, _ = self.get_best_option(self.Q, state)

        return best_option

# greedy class for metapolicies on continuous spaces
class GreedyContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, option,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=5, experiment_num=0, num_hq_iter=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.experiment_num = experiment_num

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

        # basically, a list of option starting states
        # aka the initial state and the goal states
        self.start_states = [tuple(env.all_info['ee_p'])]
        for subgoal in self.subgoals:
            self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.option = option
        # self.rewards[(start_state[x, y], subgoal_index)] = reward
        # start states consist of initial state and goal states
        # the reward function is a function over start states and the goal indices
        # I chose to use goal indices rather than goal states because when the agent reaches a goal,
        # it will not arrive at the exact goal state. So when I will need to replan after arriving at a goal,
        # which will involve adding a new start state that is also technically a goal state.
        # but I don't want to mess with the goal states, just the start states.
        self.option_rewards = self.make_option_reward_function(env)

    # do inverse kinematics to get the joint angles
    # of the 2-link reacher when the end effector
    # is at a given goal
    # note that the link lengths are assumed to be
    # 0.1 and 0.1
    # using equations from https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/
    def get_thetas_for_state(self, state):
        x = state[0]
        y = state[1]
        a1 = 0.1
        a2 = 0.1
        # there are two possible solutions to the IK problem
        # solution 1
        q2_1 = np.arccos((x**2 + y**2 - a1**2 - a2**2)/(2*a1*a2))
        q1_1 = np.arctan2(y, x) - np.arctan2(a2*np.sin(q2_1), a1 + a2*np.cos(q2_1))
        # solution 2
        q2_2 = -q2_1
        q1_2 = np.arctan2(y, x) + np.arctan2(a2*np.sin(q2_2), a1 + a2*np.cos(q2_2))

        return [[q1_1, q2_1], [q1_2, q2_2]]

    # get the reward for going from a start state to a goal state
    def get_reward_for_state(self, env, start_state, goal_state):
        joint_velocity = [0, 0]

        jp_1, jp_2 = self.get_thetas_for_state(goal_state)

        state1 = np.concatenate([
            np.cos(jp_1),
            np.sin(jp_1),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        state2 = np.concatenate([
            np.cos(jp_2),
            np.sin(jp_2),
            goal_state,
            joint_velocity,
            start_state - goal_state
        ])

        value1 = self.option.get_value(torch.Tensor(state1).float())
        value2 = self.option.get_value(torch.Tensor(state2).float())

        return max(value1, value2)

    def make_option_reward_function(self, env):
        rewards = []
        for subgoal in self.subgoals:        
            reward = {}
            for start_state in self.start_states:
                goal_state = subgoal.state
                reward[tuple(start_state)] = self.get_reward_for_state(env, start_state, goal_state).item()
            rewards.append(reward)
        return rewards

    def is_terminated(self, env, state, option):
        props = env.get_propositions(state)
        if props[option] == 1 or props[option+5] == 1:
            return True
        else:
            return False

    def get_fsa_state(self, env, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_current_propositions(threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def make_reward_function(self, task_spec):
        return task_spec.task_state_costs

    def get_closest_state(self, state):
        closest_state = None
        closest_dist = np.inf
        for s in self.start_states:
            dist = np.linalg.norm(np.array(state) - np.array(s))
            if dist < closest_dist and dist < 0.02:
                closest_dist = dist
                closest_state = s
        if closest_state is None:
            closest_state = state
        return tuple(closest_state)

    def get_option(self, env, f, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec
        
        state = self.get_closest_state(tuple(env.all_info['ee_p']))
        tm = task_spec.tm[f]

        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)

        highest_reward = -np.inf
        highest_reward_option = 0
        for i, subgoal in enumerate(self.subgoals):
            # only pick among options that lead to new states,
            # not back to the current state
            if tm[f, i] != 1:
                reward = self.option_rewards[i][state]
                if reward > highest_reward:
                    highest_reward = reward
                    highest_reward_option = i

        return highest_reward_option

if __name__ == "__main__":
    # option_path = os.path.join(os.environ['LOF_PKG_PATH'], 'external_lib', 'spinningup', 'data', 'cmd_ppo_pytorch',
                            #    'cmd_ppo_pytorch_s0', 'pyt_save', 'model.pt')

    option_path = os.path.join(os.environ['LOF_PKG_PATH'], 'experiments', 'red', 'pyt_save', 'model.pt')


    cls = Option(load_path=option_path)

    state = torch.rand(10)
    print(cls.get_action(state))
    print(cls.get_value(state))