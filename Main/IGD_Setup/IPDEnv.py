import functools

from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from Main.IGD_Setup.Action import Action  # NEU: Wichtiger Import



def _ipd_payoff(a1: Action, a2: Action) -> tuple[int, int]:
    """Standard Payoff-Matrix. Arbeitet jetzt mit Action-Enums."""
    if a1 == Action.COOPERATE and a2 == Action.COOPERATE:
        return 3, 3
    elif a1 == Action.COOPERATE and a2 == Action.DEFECT:
        return 0, 5
    elif a1 == Action.DEFECT and a2 == Action.COOPERATE:
        return 5, 0
    else:  # Action.DEFECT, Action.DEFECT
        return 1, 1


def _encode_observation(p1_action: Action, p2_action: Action) -> int:
    """Kodiert das Ergebnis der letzten Runde aus der Ich-Perspektive."""
    # Verwendet intern .value für die mathematische Berechnung
    return p1_action.value * 2 + p2_action.value


# ---------------------------------------------------------------------------

class IPDEnv(ParallelEnv):
    metadata = {"name": "ipd_v1_memory1"}

    def __init__(self, num_rounds=100):
        self.timestep = None
        self.num_rounds = num_rounds
        self.possible_agents = ["player_1", "player_2"]
        # Speichert das Tupel der letzten Aktionen als Enums
        self.last_action_pair = (Action.COOPERATE, Action.COOPERATE)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.timestep = 0
        # Neutraler Startzustand
        self.last_action_pair = (Action.COOPERATE, Action.COOPERATE)

        obs_code = _encode_observation(*self.last_action_pair)
        observations = {a: int(obs_code) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):  # Klarer Type Hint -> erwartet Integer
        """
        Nimmt Integer-Aktionen von der PettingZoo-Engine entgegen und verarbeitet sie.
        """
        if not self.agents:
            # Code für bereits beendete Umgebung
            terminations = {a: True for a in self.possible_agents}
            return ({a: 0 for a in self.possible_agents},
                    {a: 0 for a in self.possible_agents}, terminations, terminations,
                    {a: {} for a in self.possible_agents})

        self.timestep += 1

        # Wandle die Integer von PettingZoo direkt am Eingang in Enums um.
        a1_enum = Action(actions["player_1"])
        a2_enum = Action(actions["player_2"])

        # Ab hier arbeitet die gesamte interne Logik mit den sicheren und lesbaren Enums.
        r1, r2 = _ipd_payoff(a1_enum, a2_enum)
        rewards = {"player_1": r1, "player_2": r2}

        self.last_action_pair = (a1_enum, a2_enum)

        observations = {
            "player_1": _encode_observation(a1_enum, a2_enum),
            "player_2": _encode_observation(a2_enum, a1_enum)  # Perspektivwechsel
        }

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