import math
import random
from abc import ABC, abstractmethod

from typing import List, Tuple
import numpy as np


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


def calculate_grid_size(num_agents: int) -> tuple[int, int]:
    """
    Berechnet die bestmögliche, möglichst quadratische Gittergröße für eine
    gegebene Anzahl von Agenten.

    Args:
        num_agents: Die Gesamtzahl der Agenten.

    Returns:
        Ein Tupel (rows, cols), das die Gitterdimensionen darstellt.
    """
    if num_agents <= 0:
        return 0, 0
    # Beginne mit der Wurzel, um die quadratischste Form zu finden
    rows = int(math.sqrt(num_agents))
    # Finde den größten Teiler, der kleiner oder gleich der Wurzel ist
    while num_agents % rows != 0:
        rows -= 1
    cols = num_agents // rows
    return rows, cols


class SpatialGridScheme(MatchmakingScheme):
    """
    Ein Begegnungsschema, das Interaktionen auf einem 2D-Gitter verwaltet.
    Bei jedem Aufruf wird ein zufälliges Nachbarschafts-Duell zurückgegeben.
    """

    def __init__(self, neighborhood_type: str = 'moore'):
        """
        Initialisiert das Gitter-Schema.

        Args:
            neighborhood_type (str): 'moore' (8 Nachbarn) oder 'von_neumann' (4 Nachbarn).
        """
        if neighborhood_type not in ['moore', 'von_neumann', 'extended_moore']:
            raise ValueError("neighborhood_type muss 'moore' oder 'von_neumann' sein.")
        self.neighborhood_type = neighborhood_type

        # Interne Liste, um die Duelle einer Generation zu speichern
        self.match_queue = []

    def _get_neighbors(self, x: int, y: int, grid_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Findet die Koordinaten der Nachbarn für eine Zelle."""
        neighbors = []
        if self.neighborhood_type == 'moore':
            deltas = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        if self.neighborhood_type == 'von_neumann':
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:  # extended_moore
            deltas = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
            extended_deltas = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            deltas.extend(extended_deltas)

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                neighbors.append((nx, ny))
        return neighbors

    def _generate_new_generation_matches(self, grid: np.ndarray):
        """Erstellt eine neue, gemischte Liste aller Nachbarschafts-Duelle."""
        print("\n>>> Neue Generation von Gitter-Duellen wird generiert... <<<")
        matches = []
        grid_shape: tuple[int, int] = (grid.shape[0], grid.shape[1])
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                current_agent = grid[x, y]
                neighbor_coords = self._get_neighbors(x, y, grid_shape)
                for nx, ny in neighbor_coords:
                    neighbor_agent = grid[nx, ny]
                    matches.append((current_agent, neighbor_agent))

        # Mische die Liste, damit die Reihenfolge der Duelle zufällig ist
        random.shuffle(matches)
        self.match_queue = matches

    def choose_agent_pair(self, grid: np.ndarray) -> Tuple:
        """
        Wählt das nächste Nachbarschafts-Paar aus der internen Warteschlange.
        Füllt die Warteschlange neu, wenn eine Generation abgeschlossen ist.

        Args:
            grid (np.ndarray): Das 2D-Gitter, das die Agenten enthält.

        Returns:
            Ein Tupel mit zwei Agenten, die gegeneinander spielen sollen.
        """
        # Wenn die Warteschlange leer ist, starte eine neue Generation
        if not self.match_queue:
            self._generate_new_generation_matches(grid)

        # Nimm das nächste Duell aus der Warteschlange und gib es zurück
        return self.match_queue.pop()