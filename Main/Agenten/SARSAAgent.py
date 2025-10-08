import numpy as np
from Main.Agenten.BaseAgent import BaseAgent
from Main.IGD_Setup.Action import Action


# Behavioural/Target Policy: Softmax

# TODO Since SARSA and Q-Learning are almost identical, maybe create a "QLearningBasedAgent-class" from which both inherit


class SARSAAgent(BaseAgent):
    def __init__(self, n_states=4, n_actions=2, alpha=0.1, gamma=0.95, temperature=1.0, name="SARSAAgent", q_table=None):
        """
        SARSA Agent with Softmax Policy.

        Parameters
        ----------
        n_states : int
            Number of states.
        n_actions : int
            Number of actions.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor
        temperature : float
            Temperatur for Softmax Policy
        """
        super().__init__(name=name)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature


        # Initializing Q-Table
        if q_table is None:
            self.q_table = np.zeros((n_states, n_actions), dtype=float)  # Random start (50% Cooperate and 50 % Defect in every state)
            #self.q_table = np.random.rand(n_states, n_actions) # Agenten starten mit mehr Vielfalt in ihrer Policy. Werte sind zwischen 0.0 und 1.0
        else:
            self.q_table = q_table


    def _softmax(self, logits):
        """Auxiliary function: Softmax probabilities for a state vector"""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted / self.temperature)
        return exp_vals / np.sum(exp_vals)

    def choose_action(self, observation):
        """
        Selects an action with Softmax Policy based on Q values
        """
        probs = self._softmax(self.q_table[observation])
        action_index = np.random.choice(self.n_actions, p=probs)
        return Action(action_index)

    def optimize(self, obs: int, action: int, reward: float, next_obs: int, next_action: int, done: bool):
        """
        Performs a SARSA update
        """
        q_old = self.q_table[obs, action]

        if done or next_action is None:
            target = reward
        else:
            q_next = self.q_table[next_obs, next_action]
            target = reward + self.gamma * q_next

        self.q_table[obs, action] = q_old + self.alpha * (target - q_old)

    def get_policy(self):
        """
        Returns the current stochastic policy as a matrix:
        shape = (n_states, n_actions)
        """
        return np.vstack([self._softmax(self.q_table[s]) for s in range(self.n_states)])

    def train(self, experience_buffer):
        """Trains the agent with SARSA from an experience buffer"""

        # Iterate until the penultimate element
        for i in range(len(experience_buffer) - 1):
            obs, action, reward, next_obs, done = experience_buffer[i]
            # Get the next_action from the next experience
            _, next_action, _, _, _ = experience_buffer[i + 1]

            self.optimize(obs, action.value, reward, next_obs, next_action.value, done)

        # Handle the last experience
        if experience_buffer:
            obs, action, reward, next_obs, done = experience_buffer[-1]
            self.optimize(obs, action.value, reward, next_obs, None, done)