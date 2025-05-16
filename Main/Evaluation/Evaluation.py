# Pareto-Optimalität
# Kooperationsfreudigkeit von Agenten in vier sektoren aufgeteilt
# Balken-Diagramme
# Boxplots
#

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class Evaluation:
    """
    Klasse zur Erfassung und Analyse von Simulationsdaten für das iterierte Gefangenendilemma.
    Sammelt Daten über Strategievektoren, Belohnungen und Spielaktionen und bietet
    verschiedene Visualisierungsmethoden zur Analyse der Simulation.
    """

    def __init__(self):
        # Speichert Strategievektoren über Zeit
        self.strategy_history = defaultdict(list)

        # Speichert Belohnungen für jeden Agenten
        self.reward_history = defaultdict(list)

        # Speichert Aktionen beider Agenten
        self.action_history = defaultdict(list)

        # Speichert Muster/Sequenzen von Interaktionen (C-C, C-D, D-C, D-D)
        self.pattern_counts = defaultdict(lambda: defaultdict(int))

        # Speichert kumulierte Belohnungen
        self.cumulative_rewards = defaultdict(float)

        # Zählt Kooperationsraten
        self.cooperation_rates = defaultdict(list)

        # Iteration counter
        self.iterations = 0

    def capture_vector(self, agent_id, strategy_vector):
        """Erfasst den Strategievektor eines Agenten zu einem bestimmten Zeitpunkt."""
        self.strategy_history[agent_id].append(strategy_vector.copy())

        # Erfasse auch die aktuelle Kooperationswahrscheinlichkeit (Durchschnitt des Vektors)
        avg_coop = sum(strategy_vector) / len(strategy_vector)
        self.cooperation_rates[agent_id].append(avg_coop)

    def capture_reward(self, agent_id, reward):
        """Erfasst die Belohnung eines Agenten für eine Runde."""
        self.reward_history[agent_id].append(reward)
        self.cumulative_rewards[agent_id] += reward

    def capture_action(self, agent1_id, agent1_action, agent2_id, agent2_action):
        """Erfasst die Aktionen beider Agenten in einer Runde."""
        self.action_history[(agent1_id, agent2_id)].append((agent1_action, agent2_action))

        # Zählt Muster (CC, CD, DC, DD)
        pattern = ""
        pattern += "C" if agent1_action else "D"
        pattern += "C" if agent2_action else "D"
        self.pattern_counts[(agent1_id, agent2_id)][pattern] += 1

        self.iterations += 1

    def capture_info(self, agent_id, info_dict):
        """Erfasst zusätzliche Informationen für einen Agenten."""
        for key, value in info_dict.items():
            if not hasattr(self, f"info_{key}"):
                setattr(self, f"info_{key}", defaultdict(list))
            getattr(self, f"info_{key}")[agent_id].append(value)

    def plot_strategy_evolution(self, agent_id=None, title="Strategievektor-Evolution"):
        """
        Visualisiert die Evolution der Strategievektoren über die Zeit.

        Args:
            agent_id: ID des Agenten, dessen Strategie visualisiert werden soll.
                    Wenn None, werden alle Agenten dargestellt.
            title: Titel des Diagramms
        """
        plt.figure(figsize=(12, 8))

        strategy_labels = ["PCC", "PCD", "PDC", "PDD"]
        agents_to_plot = [agent_id] if agent_id else self.strategy_history.keys()

        for agent in agents_to_plot:
            strategies = np.array(self.strategy_history[agent])
            iterations = range(len(strategies))

            for i, label in enumerate(strategy_labels):
                plt.plot(iterations, strategies[:, i], label=f"{agent} - {label}")

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Kooperationswahrscheinlichkeit")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt

    def plot_rewards(self, agent_id=None, cumulative=False, rolling_window=None, title="Belohnungen pro Iteration"):
        """
        Visualisiert die Belohnungen der Agenten über die Zeit.

        Args:
            agent_id: ID des Agenten, dessen Belohnungen visualisiert werden sollen.
                    Wenn None, werden alle Agenten dargestellt.
            cumulative: Wenn True, werden kumulierte Belohnungen angezeigt.
            rolling_window: Wenn nicht None, wird ein gleitender Durchschnitt mit diesem Fenster angezeigt.
            title: Titel des Diagramms
        """
        plt.figure(figsize=(12, 6))

        agents_to_plot = [agent_id] if agent_id else self.reward_history.keys()

        for agent in agents_to_plot:
            rewards = self.reward_history[agent]
            iterations = range(len(rewards))

            if cumulative:
                cum_rewards = np.cumsum(rewards)
                plt.plot(iterations, cum_rewards, label=f"{agent} (kumuliert)")
            elif rolling_window:
                # Gleitender Durchschnitt
                rolling_mean = pd.Series(rewards).rolling(window=rolling_window).mean()
                plt.plot(iterations, rolling_mean, label=f"{agent} (gleitender Durchschnitt {rolling_window})")
            else:
                plt.plot(iterations, rewards, label=agent)

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Belohnung" if not cumulative else "Kumulierte Belohnung")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt

    def plot_cooperation_rate(self, pair=None, rolling_window=10, title="Kooperationsraten"):
        """
        Visualisiert die Rate der gegenseitigen Kooperation zwischen Agenten.

        Args:
            pair: Tupel der Form (agent1_id, agent2_id). Wenn None, werden alle Paare gezeigt.
            rolling_window: Größe des Fensters für den gleitenden Durchschnitt.
            title: Titel des Diagramms
        """
        plt.figure(figsize=(12, 6))

        pairs_to_plot = [pair] if pair else self.action_history.keys()

        for pair in pairs_to_plot:
            actions = self.action_history[pair]

            # Berechne Kooperationsrate (True = kooperieren)
            agent1_coop_rate = [1 if a[0] else 0 for a in actions]
            agent2_coop_rate = [1 if a[1] else 0 for a in actions]

            # Gleitender Durchschnitt
            agent1_rolling = pd.Series(agent1_coop_rate).rolling(window=rolling_window).mean()
            agent2_rolling = pd.Series(agent2_coop_rate).rolling(window=rolling_window).mean()
            mutual_coop = pd.Series([1 if a[0] and a[1] else 0 for a in actions]).rolling(window=rolling_window).mean()

            iterations = range(len(actions))

            plt.plot(iterations, agent1_rolling, label=f"{pair[0]} Kooperation")
            plt.plot(iterations, agent2_rolling, label=f"{pair[1]} Kooperation")
            plt.plot(iterations, mutual_coop, label="Gegenseitige Kooperation", linestyle='--')

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Kooperationsrate")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt

    def plot_pattern_distribution(self, pair=None, title="Verteilung der Interaktionsmuster"):
        """
        Visualisiert die Verteilung der Interaktionsmuster (CC, CD, DC, DD).

        Args:
            pair: Tupel der Form (agent1_id, agent2_id). Wenn None, werden alle Paare zusammengefasst.
            title: Titel des Diagramms
        """
        plt.figure(figsize=(10, 6))

        if pair:
            # Zeige nur das angegebene Paar
            patterns = self.pattern_counts[pair]
            total = sum(patterns.values())
            percentages = {k: (v / total * 100) for k, v in patterns.items()}

            plt.bar(percentages.keys(), percentages.values())
            plt.title(f"{title}: {pair[0]} vs {pair[1]}")
        else:
            # Aggregiere über alle Paare
            all_patterns = defaultdict(int)
            for pair_patterns in self.pattern_counts.values():
                for pattern, count in pair_patterns.items():
                    all_patterns[pattern] += count

            total = sum(all_patterns.values())
            percentages = {k: (v / total * 100) for k, v in all_patterns.items()}

            plt.bar(percentages.keys(), percentages.values())
            plt.title(title)

        plt.xlabel("Interaktionsmuster")
        plt.ylabel("Prozentualer Anteil")
        plt.ylim(0, 100)

        # Beschriftungen mit Prozentsätzen
        for i, (pattern, percentage) in enumerate(percentages.items()):
            plt.text(i, percentage + 1, f"{percentage:.1f}%", ha='center')

        plt.tight_layout()

        return plt

    def plot_heatmap(self, agent_id=None, title="Strategievektor-Heatmap"):
        """
        Erstellt eine Heatmap der Strategievektoren über die Zeit.

        Args:
            agent_id: ID des Agenten. Wenn None, werden alle Agenten einzeln dargestellt.
            title: Titel des Diagramms
        """
        agents_to_plot = [agent_id] if agent_id else self.strategy_history.keys()

        for agent in agents_to_plot:
            strategies = np.array(self.strategy_history[agent])

            plt.figure(figsize=(10, 6))

            sns.heatmap(strategies.T, cmap="YlGnBu", vmin=0, vmax=1,
                        yticklabels=["PCC", "PCD", "PDC", "PDD"],
                        cbar_kws={'label': 'Kooperationswahrscheinlichkeit'})

            plt.title(f"{title}: {agent}")
            plt.xlabel("Iteration")
            plt.tight_layout()

            plt.show()

    def generate_summary_report(self):
        """Erstellt einen zusammenfassenden Bericht über die Simulation."""
        report = {
            "iterations": self.iterations,
            "agents": list(set(list(self.strategy_history.keys()) + list(self.reward_history.keys()))),
            "average_rewards": {},
            "final_strategies": {},
            "cooperation_rates": {},
            "dominant_patterns": {}
        }

        # Durchschnittliche Belohnungen
        for agent, rewards in self.reward_history.items():
            report["average_rewards"][agent] = sum(rewards) / len(rewards)

        # Endstrategien
        for agent, strategies in self.strategy_history.items():
            if strategies:  # Prüfe, ob Strategien vorhanden sind
                report["final_strategies"][agent] = strategies[-1]

        # Durchschnittliche Kooperationsraten
        for pair, actions in self.action_history.items():
            agent1_coop_rate = sum(1 for a in actions if a[0]) / len(actions)
            agent2_coop_rate = sum(1 for a in actions if a[1]) / len(actions)
            mutual_coop_rate = sum(1 for a in actions if a[0] and a[1]) / len(actions)

            report["cooperation_rates"][pair] = {
                "agent1": agent1_coop_rate,
                "agent2": agent2_coop_rate,
                "mutual": mutual_coop_rate
            }

        # Dominante Interaktionsmuster
        for pair, patterns in self.pattern_counts.items():
            dominant = max(patterns.items(), key=lambda x: x[1])
            report["dominant_patterns"][pair] = {
                "pattern": dominant[0],
                "count": dominant[1],
                "percentage": dominant[1] / sum(patterns.values()) * 100
            }

        return report

    def plot_all(self, agent_id=None, pair=None):
        """Erstellt alle verfügbaren Diagramme für einen bestimmten Agenten oder ein Paar."""
        if agent_id:
            self.plot_strategy_evolution(agent_id).show()
            self.plot_rewards(agent_id).show()
            self.plot_rewards(agent_id, cumulative=True, title="Kumulierte Belohnungen").show()
            self.plot_rewards(agent_id, rolling_window=10, title="Gleitender Durchschnitt der Belohnungen").show()
            self.plot_heatmap(agent_id)

        if pair:
            self.plot_cooperation_rate(pair).show()
            self.plot_pattern_distribution(pair).show()

        if not agent_id and not pair:
            # Zeige aggregierte Diagramme
            self.plot_strategy_evolution().show()
            self.plot_rewards().show()
            self.plot_rewards(cumulative=True, title="Kumulierte Belohnungen").show()
            self.plot_cooperation_rate().show()
            self.plot_pattern_distribution().show()

    def save_to_csv(self, filename_prefix="simulation_data"):
        """Speichert alle gesammelten Daten in CSV-Dateien."""
        # Strategievektoren
        strategy_data = []
        for agent_id, vectors in self.strategy_history.items():
            for iteration, vector in enumerate(vectors):
                row = {
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "PCC": vector[0],
                    "PCD": vector[1],
                    "PDC": vector[2],
                    "PDD": vector[3]
                }
                strategy_data.append(row)

        if strategy_data:
            pd.DataFrame(strategy_data).to_csv(f"{filename_prefix}_strategies.csv", index=False)

        # Belohnungen
        reward_data = []
        for agent_id, rewards in self.reward_history.items():
            for iteration, reward in enumerate(rewards):
                row = {
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "reward": reward
                }
                reward_data.append(row)

        if reward_data:
            pd.DataFrame(reward_data).to_csv(f"{filename_prefix}_rewards.csv", index=False)

        # Aktionen
        action_data = []
        for pair, actions in self.action_history.items():
            for iteration, (action1, action2) in enumerate(actions):
                row = {
                    "agent1_id": pair[0],
                    "agent2_id": pair[1],
                    "iteration": iteration,
                    "agent1_action": "C" if action1 else "D",
                    "agent2_action": "C" if action2 else "D"
                }
                action_data.append(row)

        if action_data:
            pd.DataFrame(action_data).to_csv(f"{filename_prefix}_actions.csv", index=False)

        print(f"Daten wurden in {filename_prefix}_*.csv gespeichert.")

