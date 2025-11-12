from pathlib import Path

import numpy as np
import random

from Main.Evaluation.Evaluation import Evaluation, log_simulation_parameters, calculate_max_reward, print_results, \
    log_simulation_results
from Main.IGD_Setup.IPDEnv import IPDEnv
from Main.Matchmakingschemes.MatchmakingScheme import SpatialGridScheme, calculate_grid_size, RandomPairingScheme
from Main.SimulationSetup import GridFactory
from Main.SimulationSetup.LayoutMaps import COOP_CORE_INVASION, layout_map_defector_invasion, layout_map_blank, \
    layout_map_blank_softmax, layout_map_coop_start, layout_map_neighbor_visual_moore, \
    layout_map_neighbor_visual_extened_moore, layout_map_neighbor_visual_von_neumann

# === SIMULATION SETUP ===

# 1. Definiere alle Parameter an einem Ort
LOG_FILE = "simulation_log.md"

simulation_params = {
    "num_matches": 1000,
    "num_episodes_per_match": 1,
    "num_rounds_per_episode": 200,
    "seed": 0
}

# Definiere die Lern-Hyperparameter für die Protokollierung
learning_params = {
    "alpha": 0.05,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.9995,
    "min_epsilon": 0.001
    #"temperature": 1.0,
    #"temperature_decay": 0.9995,
    #"min_temperature": 0.001
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
    'QL': 180,
    #'QL_TFT': 9, # 18 für die Cluster + 2 extra
    'AC': 20,
} # Gesamt: 200 Agenten

# 2. Definiere die "Spezialanweisungen" für die Cluster-Platzierung
cluster_requests = [
    #{
    #    'type': 'QL_TFT', # Agententyp des Clusters
    #    'count': 1,        # Anzahl der zu erstellenden Cluster dieses Typs
    #    'neighborhood': 'moore' # (Aktuell wird immer 3x3 platziert, aber gut für die Doku)
    #},
    #{
    #    'type': 'QL_AD',  # Agententyp des Clusters
    #    'count': 1,  # Anzahl der zu erstellenden Cluster dieses Typs
    #    'neighborhood': 'moore' # Art der Nachbarschaft
    #}
]

#layout_map = GridFactory.generate_layout_with_clusters(total_composition, cluster_requests)
#layout_map = COOP_CORE_INVASION
#layout_map = layout_map_defector_invasion
#layout_map = layout_map_blank_softmax
#layout_map = layout_map_coop_start
#layout_map = layout_map_neighbor_visual_von_neumann
#layout_map = layout_map_neighbor_visual_moore
#layout_map = layout_map_neighbor_visual_extened_moore
layout_map = layout_map_blank

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
sampling_rate = 100
# === INITIALISIERE MATCHMAKING-SCHEMA ===

# Random pairing scheme
#scheme = RandomPairingScheme()

# Spatial Grid Scheme
scheme = SpatialGridScheme(neighborhood_type="moore")


evaluation.record_replay_step(grid, active_players=(None, None), current_epsilon=1.0)

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
    agent_p1, agent_p2 = scheme.choose_agent_pair(grid, match_num) # SpatialGridScheme takes grid

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
                experience_buffers[agent_id].append(experience) # Offline-Lernen (Experience-Buffer zum Speichern der Interaktionen)

                #agent_map[agent_id].train([experience]) # Online-lernen (Kein Buffer benötigt, es wird sofort gelernt)

            observations = next_observations
        env.close()

    # === 3. LERNPHASE (Batch-Update nach dem Match) ===
    #print(f"++++++Match beendet. {agent_p1.id} und {agent_p2.id} lernen jetzt...++++++")


    agent_p1.train(experience_buffers["player_1"]) # Für Offline-Lernen/Batch-Lernen wieder einklammern
    agent_p2.train(experience_buffers["player_2"]) # Für Offline-Lernen/Batch-Lernen wieder einklammern



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



    if match_num % sampling_rate == 0:
        evaluation.record_replay_step(grid, active_players=(agent_p1, agent_p2), current_epsilon=agent_p1.epsilon)

    #print_results(agent_pool, max_reward)

print("\n++++++Simulation beendet.++++++")

# === FINALE ANALYSE ===
print("\n--- Finale Strategien der Agenten im Pool ---")
print_results(agent_pool)
final_run_stats = evaluation.calculate_and_print_final_stats(agent_pool, max_reward)

if evaluation.replay_history:
    final_grid_state = evaluation.replay_history[-1]["grid"]
    cluster_results = evaluation.analyze_final_clusters(final_grid_state)
else:
    # Falls keine Replay-History gespeichert wurde, verwende das 'grid'-Objekt
    cluster_results = evaluation.analyze_final_clusters(grid)

log_simulation_results(LOG_FILE, final_run_stats=final_run_stats,final_cluster_data=cluster_results)

# === VISUALISIERUNG ===
evaluation.plot_aggregated_strategies(agent_pool, num_matches)
#evaluation.plot_strategies(agent_pool, num_matches)
evaluation.plot_aggregated_coop_rates(agent_pool, num_matches)
evaluation.plot_aggregated_rewards(agent_pool, num_matches)
#evaluation.plot_reward_by_coop_category(agent_pool, num_bins=4)

######### SAVE RESULTS #############################
OUTPUT_DIR = Path("Ergebnisse/Baseline_Setup/Datacollection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_filename = OUTPUT_DIR / f"run_data_seed_{SEED}.pkl"
evaluation.save_results(output_filename, cluster_results, final_run_stats)
#######################################################################

evaluation.render_interactive_grid_replay(cell_size=50, sampling_rate=sampling_rate)
