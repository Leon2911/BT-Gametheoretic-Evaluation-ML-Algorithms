import numpy as np
import random

from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.SARSAAgent import SARSAAgent
from Main.Agenten.WoLF_PHC_Agent import WoLFPHC
from Main.Evaluation.Evaluation import Evaluation
from Main.IGD_Setup.IPDEnv import IPDEnv
from Main.Spielfelder.MatchmakingScheme import RandomPairingScheme
from Main.IGD_Setup.IPDEnv import _ipd_payoff


# Initial strategies

# TitForTat
q_table_titfortat = [
    [9.0, 0.5],
    [0.5, 9.0],
    [9.0, 0.5],
    [0.5, 9.0]
]

# Start as Defector no matter what
q_table_defector = [
    [0.5, 9.0],
    [0.5, 9.0],
    [0.5, 9.0],
    [0.5, 9.0]
]

# Convert it to a NumPy array
q_table_defector = np.array(q_table_defector, dtype=float)
q_table_titfortat = np.array(q_table_titfortat, dtype=float)

# === SIMULATION SETUP ===

num_matches = 50
num_episodes_per_match = 5
num_rounds_per_episode = 1000

SEED = 5
random.seed(SEED)
np.random.seed(SEED)

evaluation = Evaluation()

def calculate_max_reward():
    # 1: Defect, 0: Cooperate. Somit bekommt die Variable a1 den Temptation payoff, also das Maximum
    a1, a2 = _ipd_payoff(1, 0)
    return num_matches * num_episodes_per_match * num_rounds_per_episode * a1

# Initialisiere das Matchmaking-Schema
scheme = RandomPairingScheme()

# Erstelle einen Pool mit verschiedenen Agenten
agent_pool = [
    QLearningAgent(),
    QLearningAgent(),
    QLearningAgent(),
    SARSAAgent(),
    SARSAAgent(),
    SARSAAgent(),
    #PureAgent(strategy_type="TitForTat"),
    #PureAgent(strategy_type="AlwaysDefect")
]
for agent in agent_pool:
    agent.reset_stats()

# === SIMULATIONS-SCHLEIFE ===

print("Starte Simulation mit dynamischem Matchmaking...")

for match_num in range(num_matches):

    # === 1. PAARUNGSPHASE ===
    agent_p1, agent_p2 = scheme.choose_agent_pair(agent_pool)
    agent_map = {"player_1": agent_p1, "player_2": agent_p2}

    print(f"\n--- Match {match_num + 1}/{num_matches}: {agent_p1.id} vs. {agent_p2.id} ---")

    experience_buffers = {"player_1": [], "player_2": []}

    # === 2. SPIELPHASE (Datensammlung) ===
    for episode_num in range(num_episodes_per_match):
        env = IPDEnv(num_rounds=num_rounds_per_episode)
        observations, infos = env.reset(seed=SEED)

        while env.agents:
            actions = {
                agent_id: agent_map[agent_id].choose_action(observations[agent_id])
                for agent_id in env.agents
            }
            #print(f"Observations: {observations['player_1']}")
            #print(f"Actions:      {actions['player_1']}, {actions['player_2']}")

            # Aktionen für die Statistik protokollieren
            for agent_id, agent_instance in agent_map.items():
                agent_instance.log_action(actions[agent_id])


            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            done = any(terminations.values()) or any(truncations.values())

            for agent_id in env.agents:
                agent_map[agent_id].receive_reward(rewards[agent_id])

                experience = (
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    done
                )
                experience_buffers[agent_id].append(experience)

            observations = next_observations
        env.close()

    # === 3. LERNPHASE (Batch-Update nach dem Match) ===
    print(f"++++++Match beendet. {agent_p1.id} und {agent_p2.id} lernen jetzt...++++++")

    agent_p1.train(experience_buffers["player_1"])
    agent_p2.train(experience_buffers["player_2"])

    # Ergebnisse speichern für Evaluation
    results = {}
    for agent in [agent_p1, agent_p2]:
        results[agent.id] = {
            "pi": agent.get_policy().flatten().tolist(),  # Strategievektor
            "coop_rate": agent.get_cooperation_rate() * 100,
            "reward": agent.get_total_reward()
        }
    evaluation.record(results, match_num)

    #print(f"\n--- Strategie für {agent_p1.id} ---")
    #print(agent_p1.format_strategy_vector(agent_p1.get_policy()))
    #print(f"Kooperationsrate: {agent_p1.get_cooperation_rate():.2%}")
    #print(f"Total Reward: {agent_p1.get_total_reward()}/{calculate_max_reward()}")

    #print(f"\n--- Strategie für {agent_p2.id} ---")
    #print(agent_p2.format_strategy_vector(agent_p2.get_policy()))
    #print(f"Kooperationsrate: {agent_p2.get_cooperation_rate():.2%}")
    #print(f"Total Reward: {agent_p2.get_total_reward()}/{calculate_max_reward()}")

print("\n++++++Simulation beendet.++++++")

# === FINALE ANALYSE ===
print("\n--- Finale Strategien der Agenten im Pool ---")
for agent in agent_pool:
    # Wir müssen prüfen, ob der Agent eine 'get_policy' Methode hat, die eine Matrix zurückgibt
    if isinstance(agent, (QLearningAgent, SARSAAgent, WoLFPHC)):
        #print(np.round(agent.get_policy(), 2))
        print(f"\n--- Strategie für {agent.id} ---")
        print(agent.format_strategy_vector(agent.get_policy()))
        print(f"Kooperationsrate: {agent.get_cooperation_rate():.2%}")
        print(f"Total Reward: {agent.get_total_reward()}/{calculate_max_reward()}")
    else:
        print("Agent ist kein bekannter Agententyp")

# === VISUALISIERUNG ===
evaluation.plot_strategies()
evaluation.plot_coop_rates()
evaluation.plot_rewards()