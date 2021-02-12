import torch
import os
import numpy as np
import pdb

# changed state_index to just state
class Subgoal(object):

    def __init__(self, name, prop_index, subgoal_index, state, ik_state):
        self.name = name
        # index in the overall list of props
        self.prop_index = prop_index
        # index in the list of subgoals (theoretically
        # the same as the prop_index)
        self.subgoal_index = subgoal_index
        self.state = state
        self.ik_state = ik_state

# no modifications from discrete case
class TaskSpec(object):

    def __init__(self, task_spec, tm, task_state_costs):
        self.spec = task_spec
        self.tm = tm
        self.nF = tm.shape[0]
        self.task_state_costs = task_state_costs

class RMPolicy(object):
    def __init__(self, load_path:str):
        self.ac = torch.load(load_path)

    def get_action(self, obs):
        a = self.ac.pi.mu_net(obs).detach().numpy() + 0.01*torch.normal(mean=torch.Tensor([0, 0, 0, 0])).detach().numpy()

        return a
        
    def get_value(self, state: torch.Tensor):
        return self.ac.v(state)

    def get_target_name(self):
        return self.target_name
        
    def is_terminated(self, state):
        return False

    def environment_info(self):
        pass


##########
# Option #
##########
class Option(object):
    def __init__(self, load_path:str, pick_or_place='pick', target='red_target'):
        self.ac = torch.load(load_path)
        # print(self.ac)
        # print(type(self.ac))

        self.target_name = target
        self.pick_or_place = pick_or_place
        
    def get_action(self, env_info: dict):
        state = np.concatenate([
            env_info['agent_joint_positions'],
            env_info['agent_joint_velocities'],
            env_info[self.target_name]['pos']
        ])
        
        a = self.ac.pi.mu_net(torch.from_numpy(state).float()).detach().numpy() + 0.01*torch.normal(mean=torch.Tensor([0, 0, 0, 0])).detach().numpy()
        # a = self.ac.act(torch.from_numpy(state).float())
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
        self.env = env
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

    # convert the target x,y,z positions into joint space positions
    # and preserve the initial 'agent_joint_positions' as a start state
    # return start states in the form [j0, j1, j2, j3, vj0, vj1, vj2, vj3]
    def make_start_states(self, env):
        info = env.all_info
        start_states = []
        start_vel = (0, 0, 0, 0)

        states = {
            'red_target': [-0.024184703826904297, 0.5780532360076904, 0.49106720089912415, -1.7368626594543457, -0.0010691829957067966, 1.2176384925842285, 0.7865573167800903],
                'red_goal': [-0.6136815547943115, 0.4038100242614746, -0.12854143977165222, -1.657426357269287, -0.0010327049531042576, 1.2176005840301514, 0.7865227460861206],
                'green_target': [-0.34819626808166504, 0.8475337028503418, 0.023823529481887817, -1.4239075183868408, -0.001070136670023203, 1.2176568508148193, 0.7865363359451294],
                'blue_target': [-0.47508692741394043, 0.28998875617980957, 0.03704550862312317, -1.6011130809783936, -0.0010186382569372654, 1.2176315784454346, 0.7865163087844849]
        }

        for name, data in info.items():
            if name == 'agent_joint_positions':
                start_states.append(tuple(data) + start_vel)
            elif name != 'agent_joint_velocities':
                pos = states[name][:4]
                start_states.append(tuple(pos) + start_vel)
        
        return start_states

    def make_reward_function(self, task_spec):
        return task_spec.task_state_costs

    def make_option_reward_function(self, env):
        rewards = []
        for i, (option, subgoal) in enumerate(zip(self.options, self.subgoals)):        
            reward = {}
            # start_states are in the form [j0, j1, j2, j3, vj0, vj1, vj2, vj3]
            # subgoal states are in the form [x, y, z]
            # just need to append the subgoal state to the start state to get a form
            # valid for input to the value function of the option
            for start_state in self.start_states:
                goal_state = subgoal.state
                state = start_state + tuple(goal_state)
                # print(start_state, state)
                reward[tuple(start_state)] = option.get_value(torch.Tensor(state).float()).item()
                # print(i, subgoal.name, reward[tuple(start_state)])
            rewards.append(reward)
        return rewards

    # state should be [x, y, z] I hope lol
    def is_terminated(self, env, option):
        props = env.get_current_propositions()
        if props[option] == 1:
            return True
        else:
            return False

    def get_closest_state(self, state, fk_state):
        state = state[:4] + (0, 0, 0, 0)
        closest_state = None
        closest_dist = np.inf
        for s in self.start_states:
            fk_s = self.env.fk_state(s)
            dist = np.linalg.norm(np.array(fk_state) - np.array(fk_s))
            if dist < closest_dist and dist < 0.02:
                closest_dist = dist
                closest_state = s
        if closest_state is None:
            closest_state = state
        return tuple(closest_state)

    def get_fsa_state(self, env, f, info=None, obj_name=None, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_current_propositions(info, obj_name, threshold=0.03, f=f)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])  
        # print(props, next_f)
        # print(tm[f])      
        return next_f

# LOF class for metapolicies on continuous spaces
class ContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, options,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=5, experiment_num=0, num_hq_iter=400):
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
        self.start_states = self.make_start_states(env)
        # for subgoal in self.subgoals:
            # self.start_states.append(tuple(subgoal.state))

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
        # print(self.option_rewards)
        # self.poss = self.make_poss(env.all_info['ee_p'])

        self.num_hq_iter = num_hq_iter # number of high q iterations
        self.Q = self.make_metapolicy(env, task_spec, num_iter=num_hq_iter)

    # the 'transition function' of the options, a list of dictionaries of the form
    # self.poss = [option][start state : end state]
    def make_poss(self, start_states):
        posses = []
        for subgoal in self.subgoals:
            poss = {}
            poss[tuple(start_state)] = tuple(subgoal.state)
            poss[tuple(subgoal.state)] = tuple(subgoal.state)
            posses.append(poss)
        return posses

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
        # print(num_iter)
        for k in range(num_iter):
            for f in range(self.nF):
                for i, s in enumerate(self.start_states):
                    s = tuple(s)
                    best_o = 0
                    best_o_value = -np.inf
                    # for o, poss in enumerate(self.poss):
                    for o in range(len(self.options)):
                        # if s not in poss.keys():
                            # print("{} was not in poss.keys()".format(s))
                            # poss[s] = s
                        if s not in self.option_rewards[o].keys():
                            print("{} was not in rewards[{}].keys()".format(s, o))
                            self.option_rewards[o][s] = -1.
                        
                        ns = tuple(self.subgoals[o].ik_state)[:4] + (0, 0, 0, 0)
                        props = env.get_propositions(ns, threshold=0.02)
                        p = np.where(np.array(props) == 1)
                        # print(props)
                        # print(self.tm[f])
                        # print(p, f)
                        nf = np.argmax(self.tm[f, :, p])
                        # ns = tuple(poss[s])
                        Q[(f, s, o)] = R[f]*(self.option_rewards[o][s] - 1000.0) + V[(nf, ns)]
                        # if f == self.nF - 2 and k % 100 == 0 and i == 0:
                        #     print('-------- ', k, ' o: ', o, ' props: ', props)
                        #     print(V[(nf, ns)], nf, ns)
                        #     print(R[f]*(self.option_rewards[o][s] - 1), f, s)
                        #     print(Q[(f, s, o)])
                        #     print('++++++++')
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
        # print('value function')
        # for s in self.start_states:
        #     print(V[(0, s)], s)
        return Q

    def listQ(self, Q):
        for key, value in Q.items():
            print(key, value)

    def listV(self, V):
        for key, value in V.items():
            print(key, value)

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = self.get_closest_state(tuple(env.agent.get_joint_positions()), tuple(env.agent.get_tip().get_position()))
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            # self.poss = self.make_poss(state)
            self.Q = self.make_metapolicy(env, self.task_spec, num_iter=self.num_hq_iter)
            # print('hi')

        best_option = 0
        best_option_value = -np.inf
        print("get options Q values")
        for o in range(len(self.subgoals)):
            # print(self.Q[(f, state, o)], f, o, state)
            # print(self.option_rewards[o][state])
            if self.Q[(f, state, o)] > best_option_value:
                best_option = o
                best_option_value = self.Q[(f, state, o)]

        return best_option

# LOF class for metapolicies on continuous spaces
class FSAQLearningContinuousMetaPolicy(MetaPolicyBase):
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
        self.start_states = self.make_start_states(env)
        # self.start_states = [tuple(env.all_info['ee_p'])]
        # for subgoal in self.subgoals:
            # self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.options = options
        # self.option_rewards[option][(start_state(x, y)] = reward
        self.option_rewards = self.make_option_reward_function(env)

        start_state = tuple(env.agent.get_joint_positions()[:4]) + (0, 0, 0, 0)
        self.Q = self.high_level_q_learning(env, start_state, num_hq_iter=num_hq_iter)

    def get_fsa_state_helper(self, env, state, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        props = env.get_propositions(state, threshold=0.02)
        p = np.where(np.array(props) == 1)[0][0]
        next_f = np.argmax(tm[f, :, p])        
        return next_f

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

                next_state = tuple(self.subgoals[option_index].ik_state[:4]) + (0, 0, 0, 0)
                next_f = self.get_fsa_state_helper(env, next_state, f, task_spec.tm)

                reward = task_spec.task_state_costs[f] * (self.option_rewards[option_index][current_state] - 1000.0)

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

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = self.get_closest_state(tuple(env.agent.get_joint_positions()), tuple(env.agent.get_tip().get_position()))
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            self.Q = self.high_level_q_learning(env, state, start_f=f)

        best_option = 0
        best_option_value = -np.inf
        print("get options Q values")
        for o in range(len(self.subgoals)):
            if self.Q[(f, state, o)] > best_option_value:
                best_option = o
                best_option_value = self.Q[(f, state, o)]

        return best_option


# LOF class for metapolicies on continuous spaces
class FlatQLearningContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, options,
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
        self.start_states = self.make_start_states(env)
        # self.start_states = [tuple(env.all_info['ee_p'])]
        # for subgoal in self.subgoals:
        #     self.start_states.append(tuple(subgoal.state))

        self.nF = task_spec.nF
        self.options = options
        self.option_rewards = self.make_option_reward_function(env)

        start_state = tuple(env.agent.get_joint_positions()[:4]) + (0, 0, 0, 0)
        self.Q = self.high_level_q_learning(env, start_state)


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

                next_state = tuple(self.subgoals[option_index].ik_state[:4]) + (0, 0, 0, 0)

                f = self.get_fsa_state_helper(env, next_state, f)

                if f == goal_state:
                    reward = 0
                else:
                    reward = self.option_rewards[option_index][current_state] - 1000
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
        state = self.get_closest_state(tuple(env.agent.get_joint_positions()), tuple(env.agent.get_tip().get_position()))
        if state not in self.start_states:
            self.start_states.append(state)
            self.option_rewards = self.make_option_reward_function(env)
            self.Q = self.high_level_q_learning(env, state)

        best_option, _ = self.get_best_option(self.Q, state)

        return best_option

# greedy class for metapolicies on continuous spaces
class GreedyContinuousMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env, options,
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
        self.start_states = self.make_start_states(env)

        self.nF = task_spec.nF
        self.options = options

        self.option_rewards = self.make_option_reward_function(env)

    def get_option(self, env, f, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec
        
        state = self.get_closest_state(tuple(env.agent.get_joint_positions()), tuple(env.agent.get_tip().get_position()))
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