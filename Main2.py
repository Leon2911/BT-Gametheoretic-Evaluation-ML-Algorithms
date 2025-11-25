from copy import deepcopy
from pathlib import Path
import numpy as np
import random

from Main.Agenten.BaseAgent import BaseAgent
from Main.Evaluation.Evaluation import Evaluation, log_simulation_parameters, log_simulation_results, \
    calculate_max_reward, print_results
from Main.IGD_Setup.IPDEnv import IPDEnv
from Main.Matchmakingschemes.MatchmakingScheme import SpatialGridScheme, RandomPairingScheme
from Main.SimulationSetup import GridFactory
from Main.SimulationSetup.LayoutMaps import layout_map_blank, layout_map_blank_softmax, layout_map_sarsa

# === 1. EXPERIMENT-KONFIGURATION ===
EXPERIMENT_NAME = "Setup3"

# Basis-Pfad für alle Ergebnisse
BASE_OUTPUT_DIR = Path("Ergebnisse/Datacollection")
CURRENT_OUTPUT_DIR = BASE_OUTPUT_DIR / EXPERIMENT_NAME
CURRENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Speichere Ergebnisse in: {CURRENT_OUTPUT_DIR.absolute()}")

# Log-Datei (wird im Experiment-Ordner erstellt)
LOG_FILE = CURRENT_OUTPUT_DIR / f"simulation_log{EXPERIMENT_NAME}.md"

# Anzahl der Durchläufe (Seeds)
NUM_RUNS = 10
BASE_SEED = 0

# Globale Parameter
simulation_params = {
    "num_matches": 4000,
    "num_episodes_per_match": 1,
    "num_rounds_per_episode": 200,
}

# Lern-Hyperparameter (Softmax Konfiguration)
learning_params = {
    "alpha": 0.05,
    "gamma": 0.95,
    # Softmax Parameter (gemäß Sandholm & Crites)
    #"temperature": 50.0,
    #"temperature_decay": 0.9999,
    #"min_temperature": 0.1,
    # Epsilon Parameter
    "epsilon": 1.0,
    "epsilon_decay": 0.9995,
    "min_epsilon": 0.001
}

num_matches = simulation_params["num_matches"]
num_episodes_per_match = simulation_params["num_episodes_per_match"]
num_rounds_per_episode = simulation_params["num_rounds_per_episode"]

# Korrekten System-Max-Reward berechnen (für Effizienz-Metrik)
max_system_reward = calculate_max_reward(num_matches, num_episodes_per_match, num_rounds_per_episode)

sampling_rate = 1000  # Für Replay-Snapshots

# === 2. HAUPTSCHLEIFE ÜBER ALLE SEEDS ===
for i in range(NUM_RUNS):

    current_seed = BASE_SEED + i
    print(f"\n{'=' * 60}")
    print(f"=== STARTE DURCHLAUF {i + 1}/{NUM_RUNS} (SEED: {current_seed}) ===")
    print(f"{'=' * 60}\n")

    # 1. Seed setzen
    random.seed(current_seed)
    np.random.seed(current_seed)
    BaseAgent.next_id = 0

    # 2. Evaluation neu erstellen
    evaluation = Evaluation()

    # 3. Agenten & Gitter erstellen
    #layout_map = layout_map_blank_softmax
    #layout_map = layout_map_blank # Für Epsilon-Greedy

    layout_map_for_this_run = deepcopy(layout_map_blank)

    grid, agent_pool, agent_counts = GridFactory.create_from_layout(layout_map_for_this_run)
    GRID_SIZE = grid.shape

    for agent in agent_pool:
        agent.reset_stats()

    # Initialzustand aufzeichnen (t=-1)
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
    evaluation.record(initial_results, -1)

    # 4. Matchmaking-Schema wählen
    scheme = SpatialGridScheme(neighborhood_type="moore")
    # scheme = RandomPairingScheme()

    # 5. Start-Epsilon/Temperatur für Replay holen
    start_param = 0.0
    for agent in agent_pool:
        if hasattr(agent, 'epsilon') and agent.policy == 'Epsilon-Greedy':
            start_param = agent.epsilon
            break
        elif hasattr(agent, 'temperature'):
            start_param = agent.temperature
            break

    # Replay-Snapshot bei t=0
    evaluation.record_replay_step(grid, active_players=(None, None), current_epsilon=start_param)

    # 6. Parameter loggen (nur beim ersten Lauf nötig, aber ok)
    if i == 0:
        all_params = {
            "scheme_type": scheme.__class__.__name__,
            "grid_size": GRID_SIZE if isinstance(scheme, SpatialGridScheme) else "N/A",
            **simulation_params,
            "population_composition": agent_counts,
            "learning_params": learning_params,
            "seed": current_seed
        }
        log_simulation_parameters(str(LOG_FILE), all_params)

    # === 3. SIMULATIONS-SCHLEIFE (DUELL FÜR DUELL) ===
    print(f"Starte {num_matches} Matches...")

    for match_num in range(num_matches):

        # --- GLOBALER DECAY (Für Offline-Lernen wichtig) ---
        # (Nur relevant, wenn Decay nicht in train() ist. Hier sicherheitshalber aktiv)
        if match_num > 0:
            for agent in agent_pool:
                # Epsilon Decay
                if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay'):
                    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
                # Temperature Decay (für Softmax)
                if hasattr(agent, 'temperature') and hasattr(agent, 'temperature_decay'):
                    agent.temperature = max(agent.min_temperature, agent.temperature * agent.temperature_decay)

        # A. Paarung
        agent_p1, agent_p2 = scheme.choose_agent_pair(grid, match_num)
        agent_map = {"player_1": agent_p1, "player_2": agent_p2}

        experience_buffers = {"player_1": [], "player_2": []}

        # B. Spielphase (Datensammlung)
        for episode_num in range(num_episodes_per_match):
            env = IPDEnv(num_rounds=num_rounds_per_episode)
            observations, infos = env.reset(seed=current_seed + match_num)  # Seed variieren

            while env.agents:
                actions = {
                    agent_id: agent_map[agent_id].choose_action(observations[agent_id])
                    for agent_id in env.agents
                }

                # Logging
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
                    # Offline-Lernen: Sammeln
                    experience_buffers[agent_id].append(experience)

                    # Online-Lernen: Sofort trainieren (Hier auskommentiert für Offline-Modus)
                    # agent_map[agent_id].train([experience])

                observations = next_observations
            env.close()

        # C. Lernphase (Batch-Update)
        agent_p1.train(experience_buffers["player_1"])
        agent_p2.train(experience_buffers["player_2"])

        agent_p1.log_match_played()
        agent_p2.log_match_played()

        # D. Ergebnisse aufzeichnen
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

        # E. Snapshot für Replay (Sampling)
        if match_num % sampling_rate == 0:
            curr_param = agent_p1.epsilon if hasattr(agent_p1, 'epsilon') else agent_p1.temperature
            evaluation.record_replay_step(grid, active_players=(agent_p1, agent_p2), current_epsilon=curr_param)

    # === 4. FINALE ANALYSE DES DURCHLAUFS ===
    print(f"\nDurchlauf {i + 1} beendet. Berechne Statistiken...")

    # Metriken berechnen
    # Hinweis: print_results nutzt jetzt die interne Logik für Reward/Runde
    print_results(agent_pool)

    # Berechne und hole die Stats für das Speichern
    final_run_stats = evaluation.calculate_and_print_final_stats(agent_pool, max_system_reward)

    # Cluster-Analyse
    final_grid_state = evaluation.replay_history[-1]["grid"] if evaluation.replay_history else grid
    cluster_results = evaluation.analyze_final_clusters(final_grid_state)

    current_screenshot_dir = CURRENT_OUTPUT_DIR / "Screenshots" / f"Seed_{current_seed}"
    evaluation.render_interactive_grid_replay(cell_size=50, sampling_rate=sampling_rate, auto_screenshot=True, auto_close_on_finish=True, screenshot_folder=current_screenshot_dir)

    # Ergebnisse in Log-Datei schreiben
    log_simulation_results(str(LOG_FILE), final_run_stats=final_run_stats, final_cluster_data=cluster_results)

    # === 5. SPEICHERN ===
    filename = f"run_data_seed_{current_seed}.pkl"
    full_path = CURRENT_OUTPUT_DIR / filename
    evaluation.save_results(full_path, cluster_results, final_run_stats)
    print(f"Daten gespeichert: {full_path}")

print(f"\n{'=' * 60}")
print("ALLE 10 DURCHLÄUFE ERFOLGREICH ABGESCHLOSSEN.")
print(f"Ergebnisse befinden sich in: {CURRENT_OUTPUT_DIR}")
print(f"{'=' * 60}\n")