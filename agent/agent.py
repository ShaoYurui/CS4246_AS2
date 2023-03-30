from copy import deepcopy

import numpy as np
from gym.envs.toy_text import taxi

from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import SampleSerializer

"""

    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    ### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

    ### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

    Your aim is to train a taxi agent using Value Iteration to pick up the passenger
    and drop him to his destination while maximising the rewards received.
"""


class ValueIterationAgent(object):

    def __init__(self, observation_space, action_space):
        self.policy_function = None
        self.value_policy = None
        self.env = None
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n

    def evaluate_rewards(self, problem):
        num_states = problem.observation_space.n
        num_actions = problem.action_space.n

        # Initialize rewards matrices
        rewards = np.zeros((num_states, num_actions, num_states))
        # Iterate over states, actions, and transitions
        for state in range(num_states):
            for action in range(num_actions):
                for transition in problem.P[state][action]:
                    probability, next_state, reward, done = transition
                    rewards[state, action, next_state] = reward

        return rewards

    def evaluate_transitions(self, problem):
        num_states = problem.observation_space.n
        num_actions = problem.action_space.n

        # Initialize transition matrices
        transitions = np.zeros((num_states, num_actions, num_states))

        # Iterate over states, actions, and transitions
        for state in range(num_states):
            for action in range(num_actions):
                for transition in problem.P[state][action]:
                    probability, next_state, reward, done = transition
                    transitions[state, action, next_state] = probability

                '''
                FILL ME: 
                Normalise the transition matrix, across the state and action axes.
                '''

        return transitions

    def value_iteration(self, problem, rwd_fn=None, trn_fn=None, gamma=0.9, delta=0.001):
        '''
        FILL ME : Complete with Value Iteration routine to
                      return value function and policy function
        param:
            trn_fn : shape - (|S|x|A|x|S|), trn_fn[s,a,s'] = P(s'|s,a)
            rwd_fn : shape - (|S|x|A|x|S|), rwd_fn[s,a,s'] = Reward of moving from s to s' by taking action a
            gamma : Discount factor of MDP

            Returns:
            value_fn : shape - (|S|)
            policy : shape - (|S|), dtype = int64
        '''

        value_fn = np.zeros(problem.observation_space.n)
        policy = np.zeros(problem.observation_space.n)
        if rwd_fn is None or trn_fn is None:
            rwd_fn = self.evaluate_rewards(problem)
            trn_fn = self.evaluate_transitions(problem)

        '''
        Fill up the code for Value Iteration here
        Value iteration should iterate until 
        |V_{t}(s) - V_{t+1}(s)| < delta for all states. We set delta = 0.001 and gamma = 0.9
        '''

        return value_fn, policy

    def initialize(self, env, gamma=0.001):
        self.env = env
        self.value_policy, self.policy_function = self.value_iteration(deepcopy(env), gamma)

    def step(self, state):
        action = self.policy_function[int(state)]
        return action


class TaxiAgentEnv(AgentEnv):
    def __init__(self, port: int):
        self.base_env = taxi.TaxiEnv()
        super().__init__(
            SampleSerializer(),
            self.base_env.action_space,
            self.base_env.observation_space,
            self.base_env.reward_range,
            uid=0,
            port=port,
            env=self.base_env,
        )

    def create_agent(self, **kwargs):
        agent = ValueIterationAgent(observation_space=self.base_env.observation_space,
                                    action_space=self.base_env.action_space)
        agent.initialize(self.base_env)

        return agent


def main():
    try:
        # Uncomment line 177 and comment line 178, if you have any trouble installing pygame
        # env = taxi.TaxiEnv(render_mode="ansi")
        env = taxi.TaxiEnv(render_mode="human")
        agent_env = TaxiAgentEnv(0)
        agent = agent_env.create_agent()
        state, info = env.reset(seed=2333)
        is_render = True
        total_reward = 0
        for t in range(10000):
            action = agent.step(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if is_render:
                env.render()
            total_reward += reward
            print()
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')
            print(f'total_reward     = {total_reward}')
            state = next_state
            if terminated:
                break
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
