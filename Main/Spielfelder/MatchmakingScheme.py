import random
from abc import ABC, abstractmethod

class MatchmakingScheme(ABC):
    """Abstract interface for matchmaking schemes"""

    @abstractmethod
    def choose_agent_pair(self, agents):
        """
        Select pairs of agents to compete against each other.
        :param agents: List of agent objects
        :return: List of tuples [(agent1, agent2), ...]
        """
        pass


class RandomPairingScheme(MatchmakingScheme):
    """Random pairing"""

    def choose_agent_pair(self, agents):
        pair_list = random.sample(agents, 2)
        return pair_list

class GridPairingScheme(MatchmakingScheme):

    def choose_agent_pair(self, agents):
        pass