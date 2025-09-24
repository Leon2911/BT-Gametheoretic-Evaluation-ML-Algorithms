import functools
from collections import deque

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv


class KMemoryIPDEnv(ParallelEnv):
    """
    Eine flexible Version der IPD-Umgebung für PettingZoo, die Agenten
    mit unterschiedlichen Gedächtnistiefen (memory) unterstützt.

    Die Observation eines Agenten ist die Historie der letzten k Aktionspaare,
    wobei k seine individuelle Gedächtnistiefe ist.
    Ein Aktionspaar ist immer aus der Perspektive des Agenten: (meine_aktion, gegner_aktion).
    """

    metadata = {"name": "flexible_ipd_v1"}

    def __init__(self, memories={"player_1": 1, "player_2": 1}, num_rounds=100):
        """
        Initialisiert das Environment.

        Args:
            memories (dict): Ein Dictionary, das jedem Spieler eine Gedächtnistiefe k zuweist.
                             Beispiel: {"player_1": 1, "player_2": 5}
            num_rounds (int): Die maximale Anzahl der Runden pro Episode.
        """
        super().__init__()

        # Validierung der Eingabe
        if not all(agent in memories for agent in ["player_1", "player_2"]):
            raise ValueError("Das 'memories'-Dictionary muss 'player_1' und 'player_2' enthalten.")

        self.memories = memories
        self.num_rounds = num_rounds
        self.possible_agents = ["player_1", "player_2"]
        self.agents = []

        # Die Historie speichert Tupel von (aktion_p1, aktion_p2)
        # Die maximale Länge richtet sich nach dem Agenten mit dem größten Gedächtnis
        max_mem = max(self.memories.values())
        self.history = deque(maxlen=max_mem)

        # Definiere die heterogenen Observation Spaces
        # Jeder Agent hat einen eigenen Space, der von seiner Gedächtnistiefe abhängt
        self._observation_spaces = {
            agent: MultiDiscrete(np.full((mem, 2), 2))
            for agent, mem in self.memories.items()
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Gibt den individuellen Observation Space für einen Agenten zurück."""
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Aktion: 0 für Kooperieren (Cooperate), 1 für Verraten (Defect)."""
        return Discrete(2)

    def _get_obs(self, agent):
        """
        Erstellt die Observation für einen Agenten basierend auf der Historie.
        Die Observation ist immer aus der Perspektive des Agenten.
        """
        agent_memory = self.memories[agent]

        # Nimm die letzten k relevanten Einträge aus der Historie
        # `deque` gibt automatisch die letzten `maxlen` Elemente zurück,
        # wir nehmen davon die letzten `agent_memory`
        history_slice = list(self.history)[-agent_memory:]

        # Konvertiere in ein NumPy-Array
        obs_array = np.array(history_slice, dtype=np.int8)

        # Stelle sicher, dass die Observation immer aus der Perspektive "ich, du" ist
        if agent == "player_2":
            # Für Spieler 2, tausche die Spalten (0, 1) zu (1, 0)
            obs_array = obs_array[:, [1, 0]]

        return obs_array

    def reset(self, seed=None, options=None):
        """Setzt die Umgebung zurück."""
        self.agents = self.possible_agents[:]
        self.timestep = 0

        # Fülle die Historie mit einem neutralen Anfangszustand (z.B. beide kooperieren)
        # Dies stellt sicher, dass die Observation von Anfang an die richtige Form hat
        initial_pair = (0, 0)  # Annahme: (C, C)
        self.history.clear()
        for _ in range(self.history.maxlen):
            self.history.append(initial_pair)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Führt eine Runde aus."""
        if not self.agents:
            # Wenn die Episode beendet ist, gib leere Dictionaries zurück
            return {}, {}, {}, {}, {}

        self.timestep += 1

        a1 = actions["player_1"]
        a2 = actions["player_2"]

        # Füge die aktuelle Runde zur Historie hinzu
        self.history.append((a1, a2))

        # Berechne die Belohnungen
        r1, r2 = self._ipd_payoff(a1, a2)
        rewards = {"player_1": r1, "player_2": r2}

        # Erstelle die neuen Observations basierend auf der aktualisierten Historie
        observations = {a: self._get_obs(a) for a in self.agents}

        # Prüfe, ob die Episode beendet ist
        terminations = {a: self.timestep >= self.num_rounds for a in self.agents}
        truncations = {a: False for a in self.agents}  # Kein vorzeitiger Abbruch

        if any(terminations.values()):
            self.agents = []

        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _ipd_payoff(self, a1, a2):
        """Standard IPD Payoff: T=5, R=3, P=1, S=0"""
        if a1 == 0 and a2 == 0:  # C, C
            return 3, 3
        elif a1 == 0 and a2 == 1:  # C, D
            return 0, 5
        elif a1 == 1 and a2 == 0:  # D, C
            return 5, 0
        else:  # D, D
            return 1, 1

    def render(self):
        """Einfache Textausgabe der letzten Aktion."""
        if self.history:
            last_actions = self.history[-1]
            print(f"Step {self.timestep}: P1 action={last_actions[0]}, P2 action={last_actions[1]}")
        else:
            print("Environment has not been reset yet.")


# --- Beispiel für die Verwendung ---
if __name__ == "__main__":
    # Erstelle eine Umgebung, in der Spieler 1 nur die letzte Runde sieht,
    # während Spieler 2 eine Historie der letzten 5 Runden hat.
    env = FlexibleIPDEnv(memories={"player_1": 1, "player_2": 5}, num_rounds=10)

    # Überprüfe die Observation Spaces
    print("Observation Space für Player 1 (memory=1):")
    print(env.observation_space("player_1"))
    print("\nObservation Space für Player 2 (memory=5):")
    print(env.observation_space("player_2"))

    print("\n--- Starte Test-Episode ---")
    observations, infos = env.reset()

    print("\nInitiale Observation für P1 (Form", observations["player_1"].shape, "):\n", observations["player_1"])
    print("\nInitiale Observation für P2 (Form", observations["player_2"].shape, "):\n", observations["player_2"])

    for _ in range(12):  # 12 Schritte, um Termination zu sehen
        if not env.agents:
            print("\nEpisode beendet.")
            break

        # Wähle zufällige Aktionen
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        if "player_2" in observations:  # Zeige nur die Observation von P2, da sie interessanter ist
            print("  -> Nächste Obs P2 (Form", observations["player_2"].shape, "):\n", observations["player_2"])