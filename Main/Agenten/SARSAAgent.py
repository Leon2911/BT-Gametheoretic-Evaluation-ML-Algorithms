import random

import numpy as np
from Main.Agenten.BaseAgent import BaseAgent
from Main.IGD_Setup.Action import Action


# Behavioural/Target Policy: Softmax

# TODO Since SARSA and Q-Learning are almost identical, maybe create a "QLearningBasedAgent-class" from which both inherit


class SARSAAgent(BaseAgent):
    def __init__(self, n_states=4, n_actions=2, alpha=0.05, gamma=0.95, policy="Epsilon-Greedy", name="SARSAAgent",
                 temperature=1.0, temperature_decay=0.9995, min_temperature=0.001,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.001, q_table=None):
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
        self.policy = policy

        # Softmax Hyperparameters
        self.initial_temperature = temperature
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature


        # Epsilon-Greedy Hyperparameters
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon


        # Initializing Q-Table
        if q_table is None:
            #self.q_table = np.zeros((n_states, n_actions), dtype=float)  # Random start (50% Cooperate and 50 % Defect in every state)
            self.q_table = np.random.rand(n_states, n_actions) * 0.01 # Agenten starten mit mehr Vielfalt in ihrer Policy. Werte sind zwischen 0.0 und 1.0
        else:
            self.q_table = q_table.copy()


    def reset_stats(self):
        """Resets statistics AND resets the temperature or Epsilon value to its initial value."""
        super().reset_stats()

        if self.policy == "Epsilon-Greedy":
            self.epsilon = self.initial_epsilon
        else:
            self.temperature = self.initial_temperature

    def _softmax(self, logits):
        """Auxiliary function: Softmax probabilities for a state vector"""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted / self.temperature)
        return exp_vals / np.sum(exp_vals)

    def choose_action(self, observation):
        """
        Selects an action with Softmax Policy based on Q values
        """

        if self.policy == "Epsilon-Greedy":
            if random.random() < self.epsilon:
                action_index = random.randint(0, self.n_actions - 1)
            else:
                action_index = np.argmax(self.q_table[observation])

            return Action(action_index)
        else:

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
        Returns the current policy as a matrix:
        shape = (n_states, n_actions)
        """

        if self.policy == "Epsilon-Greedy":
            greedy_policy = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                # Finde die Aktion(en) mit dem höchsten Q-Wert
                best_action_indices = np.flatnonzero(self.q_table[s] == np.max(self.q_table[s]))
                # Wähle zufällig eine der besten Aktionen bei Gleichstand
                best_action = np.random.choice(best_action_indices)
                greedy_policy[s, best_action] = 1.0
            return greedy_policy
        else:
            return np.vstack([self._softmax(self.q_table[s]) for s in range(self.n_states)])

    def train(self, experience_buffer):
        """Trains the agent with SARSA from an experience buffer"""

        # Iterate until the penultimate element
        for i in range(len(experience_buffer) - 1):
            obs, action, reward, next_obs, done = experience_buffer[i]
            # Get the next_action from the next experience
            _, next_action, _, _, _ = experience_buffer[i + 1]

            self.optimize(obs, action.value, reward, next_obs, next_action.value, done)

            if self.policy == "Epsilon-Greedy":
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            else:
                self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

        # Handle the last experience
        if experience_buffer:
            obs, action, reward, next_obs, done = experience_buffer[-1]
            self.optimize(obs, action.value, reward, next_obs, None, done)

        # Epsilon decay or Temperature decay, depending on the used policy
        if self.policy == "Epsilon-Greedy":
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        else:
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)