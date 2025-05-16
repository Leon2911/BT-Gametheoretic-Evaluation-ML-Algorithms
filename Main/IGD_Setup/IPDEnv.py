# Main/Environment/IPDEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Main.IGD_Setup.Action import Action

class IPDEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.MultiBinary(2)  # (my_last_move, opponent_last_move)
        self.action_space = spaces.Discrete(2)  # 0 = Cooperate, 1 = Defect
        self.state = np.array([0, 0])
        self.last_action_opponent = 0

    def reset(self, seed=None, options=None):
        self.state = np.array([0, 0])  # Default: Both cooperated before
        return self.state, {}

    def step(self, action):
        # Opponent: fixed tit-for-tat
        opponent_action = self.state[1]  # Spiegelt deine letzte Aktion
        self.state = np.array([action, opponent_action])

        # Belohnungsmatrix
        reward_matrix = {
            (0, 0): (3, 3),
            (0, 1): (0, 5),
            (1, 0): (5, 0),
            (1, 1): (1, 1),
        }
        if isinstance(action, np.ndarray):
            action = int(action.item())
        reward, _ = reward_matrix[(action, opponent_action)]
        done = False
        return self.state, reward, done, False, {}
