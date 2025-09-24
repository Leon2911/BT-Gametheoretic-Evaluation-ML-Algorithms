import functools
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

# Standard payoff matrix
def _ipd_payoff(a1, a2):
    if a1 == 0 and a2 == 0:
        return 3, 3
    elif a1 == 0 and a2 == 1:
        return 0, 5
    elif a1 == 1 and a2 == 0:
        return 5, 0
    else:
        return 1, 1

# Hilfsfunktion zur Kodierung der Observation
def _encode_observation(p1_action, p2_action):
    # Kodiert (C,C)->0, (C,D)->1, (D,C)->2, (D,D)->3
    return p1_action * 2 + p2_action


class IPDEnv(ParallelEnv):
    metadata = {"name": "ipd_v1_memory1"}

    def __init__(self, num_rounds=100):
        self.timestep = None
        self.num_rounds = num_rounds
        self.possible_agents = ["player_1", "player_2"]
        # Speichert das Tupel der letzten Aktionen (p1_action, p2_action)
        self.last_action_pair = (0, 0)


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.timestep = 0
        # Neutraler Startzustand (beide kooperieren)
        self.last_action_pair = (0, 0) # Oder random

        # Jeder Agent sieht den gleichen Zustand
        obs_code = _encode_observation(*self.last_action_pair)
        observations = {a: int(obs_code) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.timestep += 1
        a1, a2 = actions["player_1"], actions["player_2"]

        # Belohnungen berechnen
        r1, r2 = _ipd_payoff(a1, a2)
        rewards = {"player_1": r1, "player_2": r2}

        # Zustand für die nächste Runde speichern
        self.last_action_pair = (a1, a2)

        # Nächste Observation für beide Agenten kodieren
        observations = {
            "player_1": _encode_observation(a1, a2),
            "player_2": _encode_observation(a2, a1)
        }
        #obs_code = _encode_observation(*self.last_action_pair)
        #observations = {a: int(obs_code) for a in self.agents}

        terminations = {a: self.timestep >= self.num_rounds for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        if any(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

