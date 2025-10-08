import random

import numpy as np
from Main.Agenten.BaseAgent import BaseAgent
from Main.IGD_Setup.Action import Action


# Behavioural Policy: Softmax
# Target Policy: Greedy

# TODO Since SARSA and Q-Learning are almost identical, maybe create a "QLearningBasedAgent-class" from which both inherit

class QLearningAgent(BaseAgent):
    def __init__(self, n_states=4, n_actions=2, alpha=0.1, gamma=0.95, temperature=1, q_table=None):
        """
        Q-Learning agent with Softmax Policy

        Parameters
        ----------
        n_states : int
            Number of states
        n_actions : int
            Number of  actions
        alpha : float
            Learning rate
        gamma : float
            Discount factor
        temperature : float
            Temperatur for Softmax Policy
        """
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature

        # Initializing Q-Table
        if q_table is None:
            self.q_table = np.zeros((n_states, n_actions), dtype=float) #Random start (50% Cooperate and 50 % Defect in every state)
            #self.q_table = np.random.rand(n_states, n_actions) # Agenten starten mit mehr Vielfalt in ihrer Policy. Werte sind zwischen 0.0 und 1.0
        else:
            self.q_table = q_table # Agenten nutzen die vom Nutzer übergebene initiale Q-Table (siehe Main.py)


    def _softmax(self, logits):
        """Auxiliary function: Softmax probabilities for a state vector"""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted / self.temperature)
        return exp_vals / np.sum(exp_vals)

    def choose_action(self, observation: int) -> Action:
        """
        Selects an action with Softmax Policy based on Q values
        """
        probs = self._softmax(self.q_table[observation])
        action_index = np.random.choice(self.n_actions, p=probs)
        return Action(action_index)

    def optimize(self, obs: int, action: int, reward: float, next_obs: int, done: bool):
        """
        Performs a Q-learning update
        """
        q_old = self.q_table[obs, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_obs])
        self.q_table[obs, action] = q_old + self.alpha * (target - q_old)

    def train(self, experience_buffer):
        """Trains the agent using Q-learning from an experience buffer"""
        # Shuffle experience buffer before learning
        random.shuffle(experience_buffer)

        for obs, action, reward, next_obs, done in experience_buffer:
            # Calls its own optimize method
            self.optimize(obs, action.value, reward, next_obs, done)

    def get_policy(self):
        """
        Returns the current stochastic policy as a matrix:
        shape = (n_states, n_actions)
        """
        return np.vstack([self._softmax(self.q_table[s]) for s in range(self.n_states)])
