import numpy as np
import matplotlib.pyplot as plt


class Evaluation:

    def __init__(self):
        # Speicher für Zeitreihen: Jede Liste enthält Tupel (Zeit, Wert)
        self.strategies_over_time = {}
        self.coop_rates_over_time = {}
        self.rewards_over_time = {}

    def record(self, agents_strategies: dict, match_num: int):  # NEU: match_num wird übergeben
        """
        Speichert die Statistiken der Agenten zusammen mit dem Zeitstempel (match_num).
        """
        for agent, stats in agents_strategies.items():
            if agent not in self.strategies_over_time:
                # Initialisiere mit einem Startpunkt bei t=-1, damit die Linie bei 0 beginnt
                self.strategies_over_time[agent] = [(-1, stats["pi"])]
                self.coop_rates_over_time[agent] = [(-1, stats["coop_rate"])]
                self.rewards_over_time[agent] = [(-1, 0)]  # Start-Reward ist 0

            # Speichere den neuen Wert als (Zeit, Wert)-Tupel
            self.strategies_over_time[agent].append((match_num, stats["pi"]))
            self.coop_rates_over_time[agent].append((match_num, stats["coop_rate"]))
            self.rewards_over_time[agent].append((match_num, stats["reward"]))

    def plot_strategies(self):
        """Plottet die Entwicklung der Strategievektoren π über die Zeit."""
        if not self.strategies_over_time: return

        fig, axs = plt.subplots(len(self.strategies_over_time), 1, figsize=(10, 5 * len(self.strategies_over_time)),
                                sharex=True)
        if len(self.strategies_over_time) == 1: axs = [axs]

        labels = ["π(C|CC)", "π(C|CD)", "π(C|DC)", "π(C|DD)"]

        for ax, (agent, history) in zip(axs, self.strategies_over_time.items()):
            # Entpacke die Zeitstempel und Werte
            timestamps, values = zip(*history)
            values = np.array(values)

            for i, label in enumerate(labels):
                # KORRIGIERT: Verwende 'steps-post' um die Lücken zu füllen
                ax.plot(timestamps, values[:, i], label=label, drawstyle='steps-post')

            ax.set_title(f"Strategieentwicklung von {agent}")
            ax.set_ylabel("Kooperations-Wahrscheinlichkeit")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        # Setze den x-Achsen-Label nur für den untersten Plot
        axs[-1].set_xlabel("Match-Nummer")
        plt.tight_layout()
        plt.show()

    def plot_coop_rates(self):
        """Plottet die Entwicklung der Kooperationsraten über die Zeit."""
        if not self.coop_rates_over_time: return

        plt.figure(figsize=(10, 6))
        for agent, history in self.coop_rates_over_time.items():
            timestamps, values = zip(*history)
            # KORRIGIERT: Verwende 'steps-post'
            plt.plot(timestamps, values, label=agent, drawstyle='steps-post')

        plt.title("Kooperationsraten über die Zeit")
        plt.xlabel("Match-Nummer")
        plt.ylabel("Kooperationsrate [%]")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def plot_rewards(self):
        """Plottet die Entwicklung der Rewards über die Zeit."""
        if not self.rewards_over_time: return

        plt.figure(figsize=(10, 6))
        for agent, history in self.rewards_over_time.items():
            timestamps, values = zip(*history)
            # KORRIGIERT: Verwende 'steps-post'
            plt.plot(timestamps, values, label=agent, drawstyle='steps-post')

        plt.title("Kumulativer Reward über die Zeit")
        plt.xlabel("Match-Nummer")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()