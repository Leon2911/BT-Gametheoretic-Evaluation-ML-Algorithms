import numpy as np
import random

from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.SARSAAgent import SARSAAgent
from Main.Evaluation.Evaluation import Evaluation
from Main.IGD_Setup.IPDEnv import IPDEnv
from Main.SimulationManager import calculate_max_reward, print_results
from Main.Spielfelder.MatchmakingScheme import SpatialGridScheme, calculate_grid_size

# === INITIAL STRATEGIES FOR AGENTS ===

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
num_episodes_per_match = 2
num_rounds_per_episode = 1000
max_reward = calculate_max_reward(num_matches, num_episodes_per_match, num_rounds_per_episode)

SEED = 5
random.seed(SEED)
np.random.seed(SEED)

evaluation = Evaluation()

# === INITIALISIERE AGENTENPOOL ===
agent_pool = [
    QLearningAgent(),
    QLearningAgent(),
    QLearningAgent(),
    QLearningAgent(),
    QLearningAgent(),
    QLearningAgent(),
    SARSAAgent(),
    SARSAAgent(),
    SARSAAgent(),
    SARSAAgent(),
    SARSAAgent(),
    SARSAAgent(),
    #PureAgent(strategy_type=PureStrategy.TITFORTAT),
    #PureAgent(strategy_type=PureStrategy.ALWAYSDEFECT)
]

for agent in agent_pool:
    agent.reset_stats()

# === INITIALISIERE MATCHMAKING-SCHEMA ===

# Random pairing scheme
#scheme = RandomPairingScheme()

# Spatial Grid Scheme
scheme = SpatialGridScheme()
GRID_SIZE = calculate_grid_size(len(agent_pool))
grid = np.array(agent_pool).reshape(GRID_SIZE)

# === SIMULATIONS-SCHLEIFE ===

print("Starte Simulation mit dynamischem Matchmaking...")

for match_num in range(num_matches):

    # === 1. PAARUNGSPHASE ===
    #agent_p1, agent_p2 = scheme.choose_agent_pair(agent_pool) # RandomPairingScheme takes agent_pool
    agent_p1, agent_p2 = scheme.choose_agent_pair(grid) # SpatialGridScheme takes grid
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
            "pi": agent.get_policy().flatten().tolist(),
            "coop_rate": agent.get_cooperation_rate() * 100,
            "reward": agent.get_total_reward()
        }
    evaluation.record(results, match_num)

print("\n++++++Simulation beendet.++++++")

# === FINALE ANALYSE ===
print("\n--- Finale Strategien der Agenten im Pool ---")
print_results(agent_pool, max_reward)

# === VISUALISIERUNG ===
evaluation.plot_strategies()
evaluation.plot_coop_rates()
evaluation.plot_rewards()