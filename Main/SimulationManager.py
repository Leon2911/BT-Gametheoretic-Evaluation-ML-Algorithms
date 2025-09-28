import math

from typing import List, Dict

from Main.Agenten.BaseAgent import format_strategy_vector
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.SARSAAgent import SARSAAgent
from Main.Agenten.WoLF_PHC_Agent import WoLFPHC
from Main.IGD_Setup.Action import Action
from Main.IGD_Setup.IPDEnv import _ipd_payoff


def print_results(agent_pool, max_reward):
    # Rufe die Ranking-Funktion einmal am Anfang auf
    agent_ranks = determine_ranks(agent_pool, max_reward)

    # Wir sortieren den Pool für die Ausgabe nach dem Ranking
    sorted_agent_pool = sorted(agent_pool, key=lambda agent: agent_ranks[agent.id])

    for agent in sorted_agent_pool:
        if isinstance(agent, (QLearningAgent, SARSAAgent, WoLFPHC, PureAgent)):
            # Hole den Rang aus dem vorberechneten Dictionary
            rank = agent_ranks.get(agent.id, "N/A")

            print(f"\n--- Platz {rank}: {agent.id} ---")
            print(format_strategy_vector(agent.get_policy()))
            print(f"Kooperationsrate: {agent.get_cooperation_rate():.2%}")

            ratio = agent.get_total_reward() / max_reward
            # Hier verwende ich die von dir gewünschte Funktion zur prozentualen Darstellung
            percentage_str = f"{(ratio * 100):.1f}%".replace('.', ',')
            print(f"Total Reward: {percentage_str} des Maximums")
        else:
            print(f"Agent {agent.id} ist kein bekannter Agententyp")


def calculate_max_reward(num_matches, num_episodes_per_match, num_rounds_per_episode):
    a1, a2 = _ipd_payoff(Action.DEFECT, Action.COOPERATE)
    return num_matches * num_episodes_per_match * num_rounds_per_episode * a1


def determine_ranks(agent_pool: List, max_reward: float) -> Dict[str, int]:
    """
    Analysiert einen Pool von Agenten und weist ihnen Ränge basierend auf ihrem
    prozentualen Gesamtgewinn zu.

    Args:
        agent_pool: Eine Liste von Agenten-Objekten.
        max_reward: Der maximal mögliche Gewinn in der Simulation.

    Returns:
        Ein Dictionary, das Agenten-IDs auf ihren Rang abbildet {agent_id: rank}.
    """
    if not agent_pool or max_reward == 0:
        return {}

    # 1. Berechne die Performance für jeden Agenten
    agent_performance = []
    for agent in agent_pool:
        ratio = agent.get_total_reward() / max_reward
        agent_performance.append((ratio, agent))

    # 2. Sortiere die Agenten absteigend nach ihrer Performance
    # bei Gleichstand behält sorted() die ursprüngliche Reihenfolge bei
    sorted_agents = sorted(agent_performance, key=lambda item: item[0], reverse=True)

    # 3. Weise die Ränge zu
    ranks = {}
    for i, (ratio, agent) in enumerate(sorted_agents):
        rank = i + 1
        ranks[agent.id] = rank

    return ranks

