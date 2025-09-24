import numpy as np

from Main.Agenten.BaseAgent import BaseAgent


class WoLFPHC(BaseAgent):

    def __init__(self, n_states=4, n_actions=2, name="WoLFPHC"):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

    def choose_action(self, observation):
        """Muss in Unterklassen implementiert werden"""
        pass

    def optimize(self, obs, action, reward, next_obs, done):
        """Muss in Unterklassen implementiert werden"""
        pass

    def convert_to_mixed_strategy_vector(self, qtable):
        """Muss in Unterklassen implementiert werden"""
        pass