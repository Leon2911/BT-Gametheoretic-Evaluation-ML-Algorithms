import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any

# --- Konfiguration ---
SIMULATION_DIR = "Ergebnisse/Baseline_Setup/Datacollection"  # Der Ordner, in dem deine .pkl-Dateien liegen
NUM_MATCHES = 60000  # Muss mit deiner Main.py übereinstimmen


# === METHODE 1: DATEN LADEN ===

def load_all_runs(directory: str) -> List[Dict[str, Any]]:
    """Lädt alle .pkl-Ergebnisdateien aus einem Ordner."""
    all_runs_data = []
    file_paths = glob.glob(os.path.join(directory, "run_data_seed_*.pkl"))
    if not file_paths:
        raise FileNotFoundError(f"Fehler: Keine Ergebnisdateien in '{directory}' gefunden.")

    print(f"{len(file_paths)} Ergebnisdateien gefunden. Lade Daten...")
    for f_path in sorted(file_paths):
        with open(f_path, 'rb') as f:
            all_runs_data.append(pickle.load(f))
    return all_runs_data


# === HILFSFUNKTIONEN ZUR DATENVERARBEITUNG ===

def get_agent_type_by_id(agent_id: str) -> str:
    """Ordnet eine Agenten-ID (z.B. 'QLearningAgent_0') einer Klasse zu."""
    if agent_id.startswith("QLearningAgent"):
        return "QLearningAgent"
    if agent_id.startswith("SARSAAgent"):
        return "SARSAAgent"
    if agent_id.startswith("Pure_"):
        return "PureAgent"
    return "Unknown"


def align_time_series(histories: list, all_times: list, num_metrics=1):
    """Bringt die Daten mehrerer Agenten auf einen einheitlichen Zeitstrahl."""
    if num_metrics == 1:
        aligned_data = np.zeros((len(histories), len(all_times)))
    else:
        aligned_data = np.zeros((len(histories), len(all_times), num_metrics))

    for i, history in enumerate(histories):
        history_dict = dict(history)
        last_val = history[0][1] if history else (np.zeros(num_metrics) if num_metrics > 1 else 0)
        for j, t in enumerate(all_times):
            if t in history_dict:
                last_val = history_dict[t]
            aligned_data[i, j, ...] = last_val
    return aligned_data


# === METHODE 2: DATEN AGGREGIEREN ===

def aggregate_runs_to_means(all_runs_data: List[Dict]) -> Dict:
    """
    Nimmt die Rohdaten aller Läufe und berechnet die aggregierten Mittelwerte
    und Standardabweichungen über alle Läufe hinweg.
    """
    final_aggregated_data = {}

    # Finde den globalen Zeitstrahl (von -1 bis num_matches-1)
    all_times = sorted(
        list(set(t for run in all_runs_data for hist in run['coop_rates_over_time'].values() for t, v in hist)))
    if not all_times:
        raise ValueError("Keine Zeitreihendaten in den geladenen Dateien gefunden.")

    # Verarbeite jede Metrik (Koop-Rate, Reward, Strategie)
    for data_key in ['coop_rates_over_time', 'rewards_over_time', 'strategies_over_time']:
        num_metrics = 4 if data_key == 'strategies_over_time' else 1

        # Sammle die (bereits aggregierten) Zeitreihen von JEDEM Lauf
        data_by_type_across_runs = defaultdict(list)

        for run_data in all_runs_data:
            data_over_time = run_data[data_key]

            # Gruppiere die Rohdaten dieses Laufs nach Agententyp
            grouped_histories = defaultdict(list)
            for agent_id, history in data_over_time.items():
                agent_type = get_agent_type_by_id(agent_id)
                grouped_histories[agent_type].append(history)

            # Berechne den Mittelwert für jeden Typ innerhalb dieses einen Laufs
            for agent_type, histories in grouped_histories.items():
                aligned_data_single_run = align_time_series(histories, all_times, num_metrics)
                mean_series_single_run = np.mean(aligned_data_single_run, axis=0)
                data_by_type_across_runs[agent_type].append(mean_series_single_run)

        # Jetzt aggregiere ÜBER die Läufe
        results_for_key = {}
        for agent_type, all_run_means in data_by_type_across_runs.items():
            all_runs_array = np.array(all_run_means)
            mean_over_runs = np.mean(all_runs_array, axis=0)
            std_over_runs = np.std(all_runs_array, axis=0)

            results_for_key[agent_type] = {'mean': mean_over_runs, 'std': std_over_runs}

        final_aggregated_data[data_key] = results_for_key
        final_aggregated_data['all_times'] = all_times  # Speichere den Zeitstrahl

    # (Hier kannst du die Logik für die Aggregation der Cluster-Daten und Boxplots hinzufügen)
    # ...

    return final_aggregated_data


# === METHODE 3: FINALES PLOTTEN ===

def plot_final_graphs(aggregated_data: Dict, num_matches: int):
    """
    Nimmt das final aggregierte Dictionary und erstellt die Graphen.
    """
    all_times = aggregated_data['all_times']

    # --- PLOT 1: Kooperationsrate ---
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    for agent_type, data in aggregated_data['coop_rates_over_time'].items():
        line, = ax1.plot(all_times, data['mean'], label=f"Typ '{agent_type}'", drawstyle='steps-post')
        ax1.fill_between(all_times, data['mean'] - data['std'], data['mean'] + data['std'],
                         color=line.get_color(), alpha=0.15)
    ax1.set_title(f"Aggregierte Kooperationsrate (Mittelwert über {len(all_runs_data)} Läufe)")
    ax1.set_xlabel("Match-Nummer")
    ax1.set_ylabel("Kooperationsrate [%]")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle="--")
    ax1.set_xlim(-1, num_matches)
    ax1.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("final_plot_coop_rate.png")
    plt.show()

    # --- PLOT 2: Reward ---
    plt.figure(figsize=(12, 7))
    ax2 = plt.gca()
    for agent_type, data in aggregated_data['rewards_over_time'].items():
        line, = ax2.plot(all_times, data['mean'], label=f"Typ '{agent_type}'", drawstyle='steps-post')
        ax2.fill_between(all_times, data['mean'] - data['std'], data['mean'] + data['std'],
                         color=line.get_color(), alpha=0.15)
    ax2.set_title(f"Aggregierter kumulativer Reward (Mittelwert über {len(all_runs_data)} Läufe)")
    ax2.set_xlabel("Match-Nummer")
    ax2.set_ylabel("Total Reward")
    ax2.legend(loc='best')
    ax2.grid(True, linestyle="--")
    ax2.set_xlim(-1, num_matches)
    plt.tight_layout()
    plt.savefig("final_plot_reward.png")
    plt.show()

    # --- PLOT 3: Strategien ---
    strategy_data = aggregated_data['strategies_over_time']
    num_types = len(strategy_data)
    fig_strat, axs_strat = plt.subplots(num_types, 1, figsize=(12, 6 * num_types), sharex=True)
    if num_types == 1: axs_strat = [axs_strat]

    labels = ["π(C|CC)", "π(C|CD)", "π(C|DC)", "π(C|DD)"]
    style_map = {
        labels[0]: {'linestyle': '-', 'marker': 'o'}, labels[1]: {'linestyle': '--', 'marker': 's'},
        labels[2]: {'linestyle': ':', 'marker': '^'}, labels[3]: {'linestyle': '-.', 'marker': 'v'},
    }

    for ax, (agent_type, data) in zip(axs_strat, strategy_data.items()):
        for i, label in enumerate(labels):
            style = style_map[label]
            mean_series = data['mean'][:, i]
            std_series = data['std'][:, i]
            line, = ax.plot(all_times, mean_series, label=label,
                            drawstyle='steps-post', linestyle=style['linestyle'],
                            marker=style['marker'], markersize=2, markevery=0.2)
            ax.fill_between(all_times, mean_series - std_series, mean_series + std_series,
                            color=line.get_color(), alpha=0.15)

        ax.set_title(f"Aggregierte Strategieentwicklung '{agent_type}' (Mittelwert über {len(all_runs_data)} Läufe)")
        ax.set_ylabel("Kooperationswahrscheinlichkeit")
        ax.legend(loc='best')
        ax.grid(True, linestyle="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-1, num_matches)

    axs_strat[-1].set_xlabel("Match-Nummer")
    plt.tight_layout()
    plt.savefig("final_plot_strategies.png")
    plt.show()


# === METHODE 4 ROHDATEN PRINTEN ===

def aggregate_and_print_final_stats(all_runs_data: List[Dict]):
    """
    Aggregiert die finalen Cluster- und Kooperations-Statistiken über alle Läufe
    und gibt sie auf der Konsole aus.
    """

    # --- Dictionaries zum Sammeln der Daten aus allen Läufen ---

    # Für Metrik 1: Durchschnittlicher Reward pro Match (pro Typ)
    final_avg_rewards_by_type = defaultdict(list)
    # Für Metrik 2: Absoluter System-Reward (als % des Max)
    final_system_efficiency = []
    # Für Metrik 3: Durchschnittliche Kooperationsrate (Gesamtsystem)
    final_avg_coop_rates_global = []

    # (Zusätzliche Metriken, die wir auch sammeln)
    final_avg_coop_rates_by_type = defaultdict(list)
    final_avg_rewards_global = []  # Der "Avg. Reward pro Agent/Match"

    # (Für die Cluster-Analyse)
    final_cluster_areas = defaultdict(list)
    final_cluster_counts = defaultdict(list)
    final_max_cluster_sizes = defaultdict(list)

    # --- Datensammlung aus allen .pkl-Dateien ---
    for run_data in all_runs_data:

        # Lade die finalen Statistiken aus dem pkl-Objekt
        stats = run_data.get('final_run_statistics', {})
        if not stats:
            print("Warnung: 'final_run_statistics' in .pkl-Datei nicht gefunden. Überspringe Lauf.")
            continue

        # 1. Sammle die Kooperationsraten
        for agent_type, rate in stats.get('coop_rate_by_type', {}).items():
            final_avg_coop_rates_by_type[agent_type].append(rate)
        final_avg_coop_rates_global.append(stats.get('coop_rate_global_mean', 0.0))

        # 2. Sammle die Reward-Metriken
        for agent_type, reward in stats.get('reward_mean_by_type', {}).items():
            final_avg_rewards_by_type[agent_type].append(reward)
        final_avg_rewards_global.append(stats.get('reward_mean_global', 0.0))

        # Sammle Metrik 2 (System-Effizienz)
        final_system_efficiency.append(stats.get('reward_system_total_percent_of_max', 0.0))

        # 3. Sammle die Cluster-Daten
        cluster_data = run_data.get('final_cluster_analysis', {})
        for type_name, area in cluster_data.get('area_by_type', {}).items():
            final_cluster_areas[type_name].append(area)
        for type_name, count in cluster_data.get('count_by_type', {}).items():
            final_cluster_counts[type_name].append(count)
        for type_name, size in cluster_data.get('max_cluster_size_by_type', {}).items():
            final_max_cluster_sizes[type_name].append(size)

    # --- 3. BERECHNE MITTELWERTE UND GIB SIE AUS ---
    num_runs = len(all_runs_data)
    print("\n" + "=" * 50)
    print(f"--- AGGREGIERTE END-STATISTIKEN (Mittelwert über {num_runs} Läufe) ---")
    print("=" * 50)

    # --- METRIK 1: Durchschnittlicher Reward pro Match (pro Agententyp) ---
    print("\n**1. Durchschnittlicher Reward pro Match (pro Agententyp):**")
    if final_avg_rewards_by_type:
        sorted_rewards = sorted(final_avg_rewards_by_type.items(), key=lambda item: np.mean(item[1]), reverse=True)
        for agent_type, rewards in sorted_rewards:
            print(f"- Typ '{agent_type}': {np.mean(rewards):.2f} (± {np.std(rewards):.2f})")
    else:
        print("Keine Reward-Daten (pro Typ) gefunden.")

    # --- METRIK 2: Durchschnittliche System-Effizienz (Gesamt-Reward als % des Max) ---
    print("\n**2. Durchschnittliche System-Effizienz (Gesamt-Reward als % des theor. Maximums):**")
    if final_system_efficiency:
        print(f"- System-Effizienz: {np.mean(final_system_efficiency):.2f}% (± {np.std(final_system_efficiency):.2f})")
    else:
        print("Keine System-Effizienz-Daten gefunden.")

    # --- METRIK 3: Durchschnittliche Kooperationsrate (Gesamtsystem) ---
    print("\n**3. Durchschnittliche Kooperationsrate (Gesamtsystem, über Zeit):**")
    if final_avg_coop_rates_global:
        print(
            f"- Alle Agenten: {np.mean(final_avg_coop_rates_global):.2f}% (± {np.std(final_avg_coop_rates_global):.2f})")
    else:
        print("Keine globalen Kooperationsraten-Daten gefunden.")

    # (Detaillierte Aufschlüsselung der Koop-Rate pro Typ)
    print("\n**Durchschnittliche Kooperationsrate (pro Typ, über Zeit):**")
    if final_avg_coop_rates_by_type:
        sorted_rates = sorted(final_avg_coop_rates_by_type.items(), key=lambda item: np.mean(item[1]), reverse=True)
        for agent_type, rates in sorted_rates:
            print(f"- Typ '{agent_type}': {np.mean(rates):.2f}% (± {np.std(rates):.2f})")
    else:
        print("Keine Kooperationsraten-Daten (pro Typ) gefunden.")

    # --- CLUSTER-ANALYSE ---
    print("\n**Durchschnittliche Gesamtfläche pro Strategietyp (am Ende):**")
    if final_cluster_areas:
        sorted_areas = sorted(final_cluster_areas.items(), key=lambda item: np.mean(item[1]), reverse=True)
        for type_name, areas in sorted_areas:
            print(f"- {type_name}: {np.mean(areas):.1f}% (± {np.std(areas):.1f})")
    else:
        print("Keine Cluster-Flächen-Daten gefunden.")

    print("\n**Durchschnittliche Anzahl der Cluster pro Strategietyp (am Ende):**")
    if final_cluster_counts:
        sorted_counts = sorted(final_cluster_counts.items(), key=lambda item: np.mean(item[1]), reverse=True)
        for type_name, counts in sorted_counts:
            print(f"- {type_name}: {np.mean(counts):.1f} Cluster (± {np.std(counts):.1f})")
    else:
        print("Keine Cluster-Anzahl-Daten gefunden.")

    print("\n**Durchschnittliche Größe des größten Clusters pro Typ (am Ende):**")
    if final_max_cluster_sizes:
        sorted_sizes = sorted(final_max_cluster_sizes.items(), key=lambda item: np.mean(item[1]), reverse=True)
        for type_name, sizes in sorted_sizes:
            print(f"- {type_name}: {np.mean(sizes):.1f} Zellen (± {np.std(sizes):.1f})")
    else:
        print("Keine Max-Cluster-Größen-Daten gefunden.")

# === HAUPTFUNKTION DES ANALYSE-SKRIPTS ===
if __name__ == "__main__":
    # 1. Laden
    all_runs_data = load_all_runs(SIMULATION_DIR)

    if all_runs_data:
        # 2. Aggregieren
        final_data = aggregate_runs_to_means(all_runs_data)

        # 3. Plotten
        plot_final_graphs(final_data, NUM_MATCHES)

        # 4. Rohdaten auf Konsole printen
        aggregate_and_print_final_stats(all_runs_data)