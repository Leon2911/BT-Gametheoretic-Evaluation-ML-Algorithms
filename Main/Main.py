import numpy as np
import random

from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.SARSAAgent import SARSAAgent
from Main.Evaluation.Evaluation import Evaluation, log_simulation_parameters
from Main.IGD_Setup.IPDEnv import IPDEnv
from Main.SimulationManager import calculate_max_reward, print_results
from Main.Matchmakingschemes.MatchmakingScheme import SpatialGridScheme, calculate_grid_size, RandomPairingScheme

from Main.SimulationSetup import GridFactory
from Main.SimulationSetup.LayoutMaps import COOP_CORE_INVASION

# === SIMULATION SETUP ===

# 1. Definiere alle Parameter an einem Ort
LOG_FILE = "simulation_log.md"

simulation_params = {
    "num_matches": 3000,
    "num_episodes_per_match": 1,
    "num_rounds_per_episode": 200,
    "seed": 6
}

# Definiere die Lern-Hyperparameter für die Protokollierung
learning_params = {
    "n_states": 4,
    "n_actions": 2,
    "alpha": 0.1,
    "gamma": 0.95,
    "temperature": 1.0,
    #"temperature_decay": 0.999
}

num_matches = simulation_params["num_matches"]
num_episodes_per_match = simulation_params["num_episodes_per_match"]
num_rounds_per_episode = simulation_params["num_rounds_per_episode"]
max_reward = calculate_max_reward(num_matches, num_episodes_per_match, num_rounds_per_episode)

SEED = simulation_params["seed"]
random.seed(SEED)
np.random.seed(SEED)

evaluation = Evaluation()


# === INITIALISIERE AGENTENPOOL ===

# 1. Definiere die Gesamt-Zusammensetzung deiner Welt
total_composition = {
    'QL': 182,
    'QL_TFT': 9, # 18 für die Cluster + 2 extra
    'QL_AD': 9,
} # Gesamt: 160 Agenten

# 2. Definiere die "Spezialanweisungen" für die Cluster-Platzierung
cluster_requests = [
    {
        'type': 'QL_TFT', # Agententyp des Clusters
        'count': 1,        # Anzahl der zu erstellenden Cluster dieses Typs
        'neighborhood': 'moore' # (Aktuell wird immer 3x3 platziert, aber gut für die Doku)
    },
    {
        'type': 'QL_AD',  # Agententyp des Clusters
        'count': 1,  # Anzahl der zu erstellenden Cluster dieses Typs
        'neighborhood': 'moore' # Art der Nachbarschaft
    }
]

layout_map = GridFactory.generate_layout_with_clusters(total_composition, cluster_requests)
#layout_map = COOP_CORE_INVASION

grid, agent_pool, agent_counts = GridFactory.create_from_layout(layout_map)
GRID_SIZE = grid.shape

for agent in agent_pool:
    agent.reset_stats()

print("Speichere initialen Zustand der Agenten...")
initial_results = {}
for agent in agent_pool:
    policy = agent.get_policy()
    if policy.ndim == 2:
        coop_policy_vector = policy[:, 0]
    else:
        coop_policy_vector = policy

    initial_results[agent.id] = {
        "pi": coop_policy_vector.tolist(),
        "coop_rate": agent.get_cooperation_rate() * 100,
        "reward": agent.get_total_reward()
    }

# Record the state before any matches have been played (at time t=-1)
evaluation.record(initial_results, -1)

# === INITIALISIERE MATCHMAKING-SCHEMA ===

# Random pairing scheme
#scheme = RandomPairingScheme()

# Spatial Grid Scheme
scheme = SpatialGridScheme(neighborhood_type="moore")


#evaluation.record_replay_step(grid, active_players=(None, None))

# === SIMULATIONS-PARAMETER LOGGEN ===
all_params = {
    "scheme_type": scheme.__class__.__name__,
    "grid_size": GRID_SIZE if isinstance(scheme, SpatialGridScheme) else "N/A",
    **simulation_params, # Fügt alle Werte aus simulation_params hinzu
    "population_composition": agent_counts,
    "learning_params": learning_params
}

log_simulation_parameters(LOG_FILE, all_params)

# === SIMULATIONS-SCHLEIFE ===

print("Starte Simulation mit dynamischem Matchmaking...")
print(f"")

for match_num in range(num_matches):

    # === 1. PAARUNGSPHASE ===
    #agent_p1, agent_p2 = scheme.choose_agent_pair(agent_pool) # RandomPairingScheme takes agent_pool
    agent_p1, agent_p2 = scheme.choose_agent_pair(grid) # SpatialGridScheme takes grid
    agent_map = {"player_1": agent_p1, "player_2": agent_p2}

    #print(f"\n--- Match {match_num + 1}/{num_matches}: {agent_p1.id} vs. {agent_p2.id} ---")

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
                own_action = actions[agent_id]
                agent_map[agent_id].receive_reward(rewards[agent_id], own_action)

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
    #print(f"++++++Match beendet. {agent_p1.id} und {agent_p2.id} lernen jetzt...++++++")

    agent_p1.train(experience_buffers["player_1"])
    agent_p2.train(experience_buffers["player_2"])

    agent_p1.log_match_played()
    agent_p2.log_match_played()

    # Ergebnisse speichern für Evaluation
    results = {}
    for agent in [agent_p1, agent_p2]:
        policy = agent.get_policy()
        if policy.ndim == 2:
            coop_policy_vector = policy[:, 0]
        else:
            coop_policy_vector = policy
        results[agent.id] = {
            "pi": coop_policy_vector.tolist(),
            "coop_rate": agent.get_cooperation_rate() * 100,
            "reward": agent.get_total_reward()
        }
    evaluation.record(results, match_num)
    evaluation.record_replay_step(grid, active_players=(agent_p1, agent_p2))
    #print_results(agent_pool, max_reward)

print("\n++++++Simulation beendet.++++++")

# === FINALE ANALYSE ===
print("\n--- Finale Strategien der Agenten im Pool ---")
print_results(agent_pool, max_reward)

# === VISUALISIERUNG ===
evaluation.plot_aggregated_strategies(agent_pool, num_matches)
#evaluation.plot_strategies(agent_pool, num_matches)
evaluation.plot_aggregated_coop_rates(agent_pool, num_matches)
evaluation.plot_aggregated_rewards(agent_pool, num_matches)
evaluation.render_interactive_grid_replay(cell_size=50)