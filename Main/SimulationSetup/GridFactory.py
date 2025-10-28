import random

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Type, Tuple

from Main.Agenten.BaseAgent import BaseAgent
from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.SARSAAgent import SARSAAgent
from Main.Matchmakingschemes.MatchmakingScheme import calculate_grid_size

# === INITIAL STRATEGIES FOR AGENTS ===

# TitForTat
q_table_titfortat = [
    [9.0, 0.3],
    [0.3, 9.0],
    [9.0, 0.3],
    [0.3, 9.0]
]

# Start as Defector no matter what
q_table_defector = [
    [0.3, 9.0],
    [0.3, 9.0],
    [0.3, 9.0],
    [0.3, 9.0]
]

# Convert it to a NumPy array
q_table_defector = np.array(q_table_defector, dtype=float)
q_table_titfortat = np.array(q_table_titfortat, dtype=float)


agent_key_map = {
    # Standardeinstellung
    'QL':  {'class': QLearningAgent},
    'SA':  {'class': SARSAAgent},
    'QLE': {'class': QLearningAgent, 'params': {'policy': "Softmax"}},

    # Voreingestellte Q-Learning-Agenten
    'QL_TFT': {'class': QLearningAgent, 'params': {'q_table': q_table_titfortat}},
    'QL_AD': {'class': QLearningAgent, 'params': {'q_table': q_table_defector}},

    # Voreingestellte SARSA-Agenten
    'SA_TFT': {'class': SARSAAgent, 'params': {'q_table': q_table_titfortat}},
    'SA_AD': {'class': SARSAAgent, 'params': {'q_table': q_table_defector}},

    # Reine Strategien
    'TFT': {'class': PureAgent, 'params': {'strategy_type': PureStrategy.TITFORTAT}},
    'AD':  {'class': PureAgent, 'params': {'strategy_type': PureStrategy.ALWAYSDEFECT}},
    'AC': {'class': PureAgent, 'params': {'strategy_type': PureStrategy.ALWAYSCOOPERATE}},
}

def create_from_layout(layout_map: List[List[str]]) -> Tuple[np.ndarray, List, Counter]:
    """
    Erstellt ein Gitter und einen Agenten-Pool basierend auf einer vordefinierten Landkarte.

    Args:
        layout_map: Eine 2D-Liste von Strings, die die Agententypen repräsentieren.
        agent_key_map: Ein Dictionary, das die String-Kürzel auf die Agenten-Klassen
                       und deren Initialisierungsparameter abbildet.

    Returns:
        Ein Tupel (grid, agent_pool), das das 2D-Gitter und die flache Liste der Agenten enthält.
    """
    # 1. Setze den globalen ID-Zähler zurück. Wichtig für reproduzierbare Experimente!
    BaseAgent.next_id = 0

    # 2. Leite die Gittergröße ab und erstelle leere Datenstrukturen
    grid_size = (len(layout_map), len(layout_map[0]))
    grid = np.empty(grid_size, dtype=object)
    agent_pool = []

    # 3. Iteriere durch das Gitter und erstelle jeden Agenten an seiner Position
    for r in range(grid_size[0]):  # r für row (Zeile)
        for c in range(grid_size[1]):  # c für column (Spalte)
            agent_key = layout_map[r][c]

            if agent_key not in agent_key_map:
                raise ValueError(f"Unbekannter Agenten-Schlüssel '{agent_key}' in der layout_map.")

            # Hole die Konfiguration für den benötigten Agententyp
            config = agent_key_map[agent_key]
            agent_class = config['class']
            params = config.get('params', {})

            # Erstelle die Agenten-Instanz GENAU HIER!
            # BaseAgent.next_id wird jetzt in der Reihenfolge 0, 1, 2, ... hochgezählt.
            new_agent = agent_class(**params)

            # Platziere den neuen Agenten
            grid[r, c] = new_agent
            agent_pool.append(new_agent)

    # 4. Berechne die Zusammensetzung für das Logging (optional, aber nützlich)
    flat_map = [key for row in layout_map for key in row]
    population_composition = Counter(flat_map)

    print(f"Erstelle ein {grid_size}-Gitter nach Plan: {population_composition}")

    return grid, agent_pool, population_composition

def generate_layout_with_clusters(total_composition: dict, cluster_requests: list) -> list[list[str]]:
    """
    Generiert prozedural eine layout_map basierend auf einer Gesamtpopulation
    und Anfragen zur Cluster-Bildung.

    Args:
        total_composition: Ein Dict, das die Gesamtzahl jedes Agententyps definiert.
                           z.B. {'QL': 100, 'TFT': 20, 'AD': 4}
        cluster_requests: Eine Liste von Dicts, die die zu platzierenden Cluster beschreiben.
                          z.B. [{'type': 'TFT', 'count': 2, 'neighborhood': 'moore'}]

    Returns:
        Eine 2D-Liste (layout_map), die an create_from_layout übergeben werden kann.
    """
    # 1. Berechne die Gesamtgröße des Gitters
    num_agents = sum(total_composition.values())
    grid_size = calculate_grid_size(num_agents)
    rows, cols = grid_size

    # Erstelle ein leeres Gitter mit Platzhaltern
    layout = np.full(grid_size, None, dtype=object)

    agents_to_place = Counter(total_composition)

    # 2. Platziere die angeforderten Cluster
    print("Platziere vordefinierte Cluster...")
    for request in cluster_requests:
        cluster_type = request['type']
        cluster_count = request['count']

        for _ in range(cluster_count):
            # Finde einen validen, freien Platz für einen 3x3 Cluster
            for attempt in range(100):  # Sicherheits-Loop gegen Endlosschleifen
                # Wähle einen zufälligen Mittelpunkt (nicht am Rand)
                center_r = random.randint(1, rows - 2)
                center_c = random.randint(1, cols - 2)

                # Prüfe, ob der 3x3 Bereich frei ist
                if np.all(layout[center_r - 1:center_r + 2, center_c - 1:center_c + 2] == None):
                    # Platziere den Cluster
                    layout[center_r - 1:center_r + 2, center_c - 1:center_c + 2] = cluster_type
                    agents_to_place[cluster_type] -= 9  # Ziehe 9 Agenten vom Zähler ab
                    print(f"- Platziere '{cluster_type}' Cluster bei ({center_r}, {center_c})")
                    break
            else:  # Wenn nach 100 Versuchen kein Platz gefunden wurde
                raise RuntimeError(
                    f"Konnte nach 100 Versuchen keinen Platz für einen '{cluster_type}' Cluster finden.")

    # 3. Fülle den Rest des Gitters mit den verbleibenden Agenten
    remaining_agents = list(agents_to_place.elements())
    random.shuffle(remaining_agents)

    for r in range(rows):
        for c in range(cols):
            if layout[r, c] is None:
                if not remaining_agents:
                    raise ValueError("Mehr Gitterplätze als Agenten in der Komposition definiert.")
                layout[r, c] = remaining_agents.pop()

    return layout.tolist()

class GridFactory:
    """
    Eine Factory-Klasse, die für die Erstellung und Konfiguration
    des Agenten-Gitters zuständig ist.
    """

