import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

import pygame

from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.Agenten.PureAgent import PureAgent
from Main.Agenten.QLearningAgent import QLearningAgent
from Main.Agenten.SARSAAgent import SARSAAgent

CELL_SIZE = 20

# Farbpalettendefinitionen
BLUE = (50, 150, 255)  # Kooperativ (TFT-ähnlich)
RED = (255, 50, 50)  # Defektiv (ALLD-ähnlich)
GREEN = (50, 255, 50)  # Rein Kooperativ (ALLC)
MINT_GREEN = (152, 251, 152) # Für starke Kooperatoren
CYAN = (0, 200, 200) # Kooperiert nur, wenn beide letzte Runde kooperiert haben
PURPLE = (200, 100, 255) # Win-Stay, Lose-Shift (Pavlov)
ORANGE = (255, 140, 0)
YELLOW = (255, 200, 50)  # Gemischte/Lernende Strategie
GREY = (128, 128, 128)  # Unbekannt oder neutral


def lerp_color(color1: tuple, color2: tuple, factor: float) -> tuple[int, int, int]:
    """
    Lineare Interpolation zwischen zwei RGB-Tupeln.
    Wandelt intern in NumPy-Arrays für die Berechnung um.
    """
    # Konvertiere die Input-Tupel in NumPy-Arrays
    c1 = np.array(color1)
    c2 = np.array(color2)

    result = c1 * (1 - factor) + c2 * factor

    return tuple(result.astype(int))


def get_agent_color_spectrum(agent) -> tuple[int, int, int]:
    """
    Analysiert die Strategie eines Agenten und weist eine Farbe aus einem
    kontinuierlichen Spektrum für die jeweilige Kategorie zu.
    """
    if isinstance(agent, PureAgent):
        if agent.strategy_type == PureStrategy.ALWAYSDEFECT:
            return RED
        elif agent.strategy_type == PureStrategy.TITFORTAT:
            return BLUE
        elif agent.strategy_type == PureStrategy.ALWAYSCOOPERATE:
            return GREEN
        else:
            return PURPLE #GrimTrigger, Random etc

    elif isinstance(agent, (QLearningAgent, SARSAAgent)):
        policy = agent.get_policy()
        if policy.ndim == 2:
            coop_policy = policy[:, 0]
        else:
            coop_policy = policy
        p_cc, p_cd, p_dc, p_dd = coop_policy

        # --- KATEGORISIERUNG MIT FARBSPEKTRUM ---

        # Tit-for-Tat-Spektrum (BLAU)
        if p_cc > 0.8 and p_cd < 0.2 and p_dc > 0.8 and p_dd < 0.2:
            # Intensität basiert auf der Übereinstimmung mit dem TFT-Muster.
            intensity = (p_cc + (1 - p_cd) + p_dc + (1 - p_dd)) / 4.0
            return lerp_color(YELLOW, BLUE, intensity)

        # Win-Stay-Lose-Shift-Spektrum (PURPLE)
        elif p_cc > 0.8 and p_cd < 0.2 and p_dc < 0.2 and p_dd > 0.2:
            # Intensität basiert auf der Übereinstimmung mit dem WSLS-Muster.
            intensity = (p_cc + (1 - p_cd) + (1 - p_dc) + p_dd) / 4.0
            return lerp_color(YELLOW, PURPLE, intensity)

        # Vorsichtiger-Kooperator-Spektrum (CYAN)
        elif p_cc > 0.8 and p_cd < 0.2 and p_dc < 0.2 and p_dd < 0.2:
            # Intensität basiert darauf, wie stark die TFT-ähnliche Reaktion ist.
            intensity = (p_cc + (1 - p_cd)) / 2.0
            return lerp_color(YELLOW, CYAN, intensity)

        elif np.sum(coop_policy > 0.8) == 2 and np.sum(coop_policy < 0.2) == 2:
            # Die Intensität spiegelt wider, wie extrem die Polarisierung ist.
            # (Werte nahe 1 und 0 geben eine höhere Intensität)
            intensity = np.mean([p if p > 0.5 else 1-p for p in coop_policy])
            return lerp_color(YELLOW, ORANGE, float(intensity))

        # Allgemeines Kooperations-Spektrum (GRÜN)
        # Wenn mindestens 2 Werte > 0.8 sind
        elif np.sum(coop_policy > 0.8) >= 2:
            # Die Intensität ist der Durchschnitt der Kooperations-Wahrscheinlichkeiten
            intensity = np.mean(coop_policy)
            return lerp_color(YELLOW, GREEN, intensity)

        # Allgemeines Defektor-Spektrum (ROT)
        # Wenn mindestens 2 Werte < 0.2 sind
        elif np.sum(coop_policy < 0.2) >= 2:
            # Die Intensität ist die durchschnittliche Defektions-Wahrscheinlichkeit
            intensity = 1.0 - np.mean(coop_policy)
            return lerp_color(YELLOW, RED, intensity)

        # Fallback 1 für gemischte Strategien
        else:
            return YELLOW

    # Fallback 2 für unbekannte Agententypen
    return GREY


def _group_by_type(data_dict, agent_pool):
    """Hilfsfunktion, um Agenten-Daten nach Typ zu gruppieren."""
    grouped_data = defaultdict(list)
    agent_type_map = {agent.id: agent.__class__.__name__ for agent in agent_pool}

    for agent_id, history in data_dict.items():
        agent_type = agent_type_map[agent_id]
        grouped_data[agent_type].append(history)

    return grouped_data

def log_simulation_parameters(filepath: str, params: dict):
    """
    Protokolliert alle Simulationsparameter auf der Konsole und in einer persistenten Datei.

    Args:
        filepath (str): Der Pfad zur Log-Datei (z.B. "simulation_log.md").
        params (dict): Ein Dictionary, das alle relevanten Parameter enthält.
    """
    # 1. Zeitstempel für diesen Simulationslauf erstellen
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Den Log-Eintrag als formatierten String (Markdown) erstellen
    log_entry = f"## Simulationslauf vom: {timestamp}\n\n"

    # Globale Simulations-Parameter
    log_entry += "### Globale Parameter\n"
    log_entry += f"- **Begegnungsschema:** `{params['scheme_type']}`\n"
    if 'grid_size' in params:
        log_entry += f"- **Gittergröße:** `{params['grid_size']}`\n"
    log_entry += f"- **Anzahl Matches/Duelle:** `{params['num_matches']}`\n"
    log_entry += f"- **Runden pro Episode:** `{params['num_rounds_per_episode']}`\n"
    log_entry += f"- **Zufalls-Seed:** `{params['seed']}`\n\n"

    # Agenten-Population
    # Agenten-Population
    log_entry += "### Agenten-Population\n"
    total_agents = 0
    # Greife auf den neuen Schlüssel zu und iteriere über das einfache Counter-Objekt
    for agent_name, count in params['population_composition'].items():
        if count > 0:
            log_entry += f"- **{agent_name}:** `{count}`\n"
            total_agents += count
    log_entry += f"- **Gesamt:** `{total_agents}` Agenten\n\n"

    # Lern-Hyperparameter der lernenden Agenten
    log_entry += "### Lern-Hyperparameter\n"
    for key, value in params['learning_params'].items():
        log_entry += f"- **{key}:** `{value}`\n"

    log_entry += "\n---\n\n"  # Trennlinie für den nächsten Lauf

    # 3. Den Log-Eintrag auf der Konsole ausgeben
    print("--- START: Simulations-Parameter ---")
    print(log_entry.replace("### ", "").replace("## ", ""))  # Ohne Markdown-Formatierung für die Konsole
    print("--- ENDE: Simulations-Parameter ---")

    # 4. Den Log-Eintrag an die Datei anhängen
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except IOError as e:
        print(f"Fehler beim Schreiben der Log-Datei: {e}")



class Evaluation:

    def __init__(self):
        # Die Datenstruktur bleibt gleich: {agent_id: [(time, value), ...]}
        self.strategies_over_time = defaultdict(list)
        self.coop_rates_over_time = defaultdict(list)
        self.rewards_over_time = defaultdict(list)

        self.replay_history = []

    def record_replay_step(self, grid, active_players):
        """Speichert den Gitterzustand und die aktiven Spieler für einen Replay-Schritt."""
        self.replay_history.append({
            "grid": deepcopy(grid),
            "players": (active_players[0].id, active_players[1].id)
        })

    def record(self, agents_strategies: dict, match_num: int):
        for agent_id, stats in agents_strategies.items():
            if agent_id not in self.strategies_over_time:
                # Initialisiere die Liste für diesen Agenten
                self.strategies_over_time[agent_id] = []
                self.coop_rates_over_time[agent_id] = []
                self.rewards_over_time[agent_id] = []

            # Füge das (Zeit, Wert)-Tupel hinzu
            self.strategies_over_time[agent_id].append((match_num, stats["pi"]))
            self.coop_rates_over_time[agent_id].append((match_num, stats["coop_rate"]))
            self.rewards_over_time[agent_id].append((match_num, stats["reward"]))

    def plot_strategies(self, agent_pool, num_matches):
        """Plottet die Entwicklung der Strategievektoren π für jeden einzelnen Agenten."""
        if not self.strategies_over_time: return

        # Sortiere die Agenten-IDs, um eine konsistente Plot-Reihenfolge zu haben
        sorted_agent_ids = sorted(self.strategies_over_time.keys())

        num_agents = len(sorted_agent_ids)
        fig, axs = plt.subplots(num_agents, 1, figsize=(12, 6 * num_agents), sharex=True)
        if num_agents == 1: axs = [axs]

        # Definiere ein Dictionary, das jeder Strategie einen einzigartigen Stil zuweist
        labels = ["π(C|CC)", "π(C|CD)", "π(C|DC)", "π(C|DD)"]
        style_map = {
            labels[0]: {'linestyle': '-', 'marker': 'o', 'color': 'blue'},
            labels[1]: {'linestyle': '--', 'marker': 's', 'color': 'red'},
            labels[2]: {'linestyle': ':', 'marker': '^', 'color': 'green'},
            labels[3]: {'linestyle': '-.', 'marker': 'v', 'color': 'purple'},
        }

        for ax, agent_id in zip(axs, sorted_agent_ids):
            history = self.strategies_over_time[agent_id]

            # Entpacke die Zeitstempel und Werte
            timestamps, values = zip(*history)
            values = np.array(values)

            for i, label in enumerate(labels):
                style = style_map[label]
                # Füge die Stil-Argumente zum plot-Befehl hinzu
                ax.plot(timestamps, values[:, i],
                        label=label,
                        drawstyle='steps-post',
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        color=style['color'],
                        markersize=4,
                        markevery=0.1)  # Setze Marker nur alle 10% der Datenpunkte

            ax.set_title(f"Strategieentwicklung von {agent_id}")
            ax.set_ylabel("Kooperations-Wahrscheinlichkeit")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_ylim(-0.05, 1.05)
            # Setze die Achsengrenzen explizit für eine korrekte Darstellung
            ax.set_xlim(0, num_matches)

        axs[-1].set_xlabel("Match-Nummer")
        plt.tight_layout()
        plt.show()


    def plot_aggregated_strategies(self, agent_pool, num_matches):
        """Plottet die DURCHSCHNITTLICHE Strategieentwicklung pro Agenten-TYP."""
        if not self.strategies_over_time: return

        grouped_strategies = _group_by_type(self.strategies_over_time, agent_pool)
        num_types = len(grouped_strategies)

        fig, axs = plt.subplots(num_types, 1, figsize=(12, 6 * num_types), sharex=True)
        if num_types == 1: axs = [axs]


        # Definiere ein Dictionary, das jeder Strategie einen einzigartigen Stil zuweist
        labels = ["π(C|CC)", "π(C|CD)", "π(C|DC)", "π(C|DD)"]
        style_map = {
            labels[0]: {'linestyle': '-', 'marker': 'o'},  # Durchgezogen, Kreis
            labels[1]: {'linestyle': '--', 'marker': 's'},  # Gestrichelt, Quadrat
            labels[2]: {'linestyle': ':', 'marker': '^'},  # Gepunktet, Dreieck
            labels[3]: {'linestyle': '-.', 'marker': 'v'},  # Strich-Punkt, Dreieck unten
        }


        for ax, (agent_type, histories) in zip(axs, grouped_strategies.items()):
            # ... (deine Logik zur Datenaufbereitung bleibt unverändert) ...
            all_times = sorted(list(set(t for history in histories for t, v in history)))
            aligned_strategies = np.zeros((len(histories), len(all_times), 4))
            for i, history in enumerate(histories):
                history_dict = dict(history)
                last_val = history[0][1] if history else np.zeros(4)
                for j, t in enumerate(all_times):
                    if t in history_dict: last_val = history_dict[t]
                    aligned_strategies[i, j, :] = last_val
            mean_strategy = np.mean(aligned_strategies, axis=0)

            for i, label in enumerate(labels):
                style = style_map[label]
                # Füge die Stil-Argumente zum plot-Befehl hinzu
                ax.plot(all_times, mean_strategy[:, i],
                        label=label,
                        drawstyle='steps-post',
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markersize=4,  # Mache die Marker etwas kleiner
                        markevery=0.1)  # Setze nur alle 10% der Datenpunkte einen Marker

            ax.set_title(f"Durchschnittliche Strategieentwicklung für Typ '{agent_type}' ({len(histories)} Agenten)")
            ax.set_ylabel("Kooperations-Wahrscheinlichkeit")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, num_matches)

        axs[-1].set_xlabel("Match-Nummer")
        plt.tight_layout()
        plt.show()

    def plot_aggregated_coop_rates(self, agent_pool, num_matches):
        """Plottet die DURCHSCHNITTLICHE Kooperationsrate pro Agenten-TYP."""
        if not self.coop_rates_over_time: return

        grouped_rates = _group_by_type(self.coop_rates_over_time, agent_pool)

        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        for agent_type, histories in grouped_rates.items():
            # Finde alle einzigartigen Zeitstempel für diese Gruppe
            all_times = sorted(list(set(t for history in histories for t, v in history)))

            # Erstelle ein Array, um die Werte aller Agenten dieses Typs zu speichern
            aligned_rates = np.zeros((len(histories), len(all_times)))

            for i, history in enumerate(histories):
                history_dict = dict(history)
                last_val = history[0][1] if history else 0
                for j, t in enumerate(all_times):
                    if t in history_dict:
                        last_val = history_dict[t]
                    aligned_rates[i, j] = last_val

            # Berechne Durchschnitt und Standardabweichung
            mean_rates = np.mean(aligned_rates, axis=0)
            std_rates = np.std(aligned_rates, axis=0)

            # Zeichne die Durchschnittslinie
            line, = ax.plot(all_times, mean_rates, label=f"Typ '{agent_type}' ({len(histories)} Agenten)",
                            drawstyle='steps-post')

            # Füge das Fehlerband (Standardabweichung) hinzu
            ax.fill_between(all_times,
                            mean_rates - std_rates,
                            mean_rates + std_rates,
                            color=line.get_color(), alpha=0.2)

        ax.set_title("Durchschnittliche Kooperationsraten über die Zeit")
        ax.set_xlabel("Match-Nummer")
        ax.set_ylabel("Kooperationsrate [%]")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0, num_matches)
        plt.tight_layout()
        plt.show()

    def plot_aggregated_rewards(self, agent_pool, num_matches):
        """Plottet den DURCHSCHNITTLICHEN kumulativen Reward pro Agenten-TYP."""
        if not self.rewards_over_time: return

        grouped_rewards = _group_by_type(self.rewards_over_time, agent_pool)

        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        for agent_type, histories in grouped_rewards.items():
            all_times = sorted(list(set(t for history in histories for t, v in history)))
            aligned_rewards = np.zeros((len(histories), len(all_times)))

            for i, history in enumerate(histories):
                history_dict = dict(history)
                last_val = history[0][1] if history else 0
                for j, t in enumerate(all_times):
                    if t in history_dict:
                        last_val = history_dict[t]
                    aligned_rewards[i, j] = last_val

            mean_rewards = np.mean(aligned_rewards, axis=0)
            std_rewards = np.std(aligned_rewards, axis=0)

            line, = ax.plot(all_times, mean_rewards, label=f"Typ '{agent_type}' ({len(histories)} Agenten)",
                            drawstyle='steps-post')

            ax.fill_between(all_times,
                            mean_rewards - std_rewards,
                            mean_rewards + std_rewards,
                            color=line.get_color(), alpha=0.2)

        ax.set_title("Durchschnittlicher kumulativer Reward über die Zeit")
        ax.set_xlabel("Match-Nummer")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(0, num_matches)
        plt.tight_layout()
        plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++ RENDERING ++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def render_interactive_grid_replay(self, cell_size=30):
        """
        Startet ein interaktives Dashboard mit präzisem Layout und allen Features.
        """
        if not self.replay_history:
            print("Keine Replay-Historie zum Anzeigen vorhanden.")
            return

        pygame.init()
        pygame.key.set_repeat(200, 35)

        grid_shape = self.replay_history[0]["grid"].shape

        # --- LAYOUT-BERECHNUNG ---
        heatmap_cell_size = 15
        heatmap_height = grid_shape[0] * heatmap_cell_size
        heatmap_width = grid_shape[1] * heatmap_cell_size
        colorbar_height = 10
        title_space = 25
        label_space = 20
        spacing_between_heatmaps = 40
        spacing_between_bottom = 10

        left_panel_width = heatmap_width + 50  # 10px Rand + 40px für die Legende
        left_panel_height = (heatmap_height + colorbar_height + title_space + label_space) * 3 + spacing_between_heatmaps + spacing_between_bottom

        grid_panel_width = grid_shape[1] * cell_size
        grid_panel_height = grid_shape[0] * cell_size
        right_panel_width = 300

        screen_width = left_panel_width + grid_panel_width + right_panel_width


        main_content_height = max(left_panel_height, grid_panel_height, 400)  # Mindesthöhe von 400px
        info_bar_height = 50
        screen_height = main_content_height + info_bar_height

        screen_size = (screen_width, screen_height)

        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Interaktives Analyse-Dashboard")
        font = pygame.font.Font(None, 28)
        id_font = pygame.font.Font(None, int(cell_size * 0.6))
        clock = pygame.time.Clock()

        current_step = 0
        running = True
        scroll_offset = 0
        scroll_speed = 20

        panel_y_start = 40
        panel_line_height = 20

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        current_step = min(current_step + 1, len(self.replay_history) - 1)
                    elif event.key == pygame.K_LEFT:
                        current_step = max(current_step - 1, 0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        scroll_offset = max(0, scroll_offset - scroll_speed)
                    elif event.button == 5:  # Mausrad nach unten
                        # Berechne die maximale Scroll-Position
                        num_agents = len(grid_to_draw.flatten())
                        total_content_height = num_agents * panel_line_height
                        visible_panel_height = screen.get_height() - panel_y_start - info_bar_height

                        # Der maximale Offset ist die Gesamthöhe minus der sichtbaren Höhe
                        max_scroll = max(0, total_content_height - visible_panel_height)

                        # Stelle sicher, dass der neue Offset das Maximum nicht überschreitet
                        scroll_offset = min(scroll_offset + scroll_speed, max_scroll)

            screen.fill((20, 20, 20))
            current_data = self.replay_history[current_step]
            grid_to_draw = current_data["grid"]
            active_players = current_data["players"]

            # --- ZEICHNEN DER 4 PANELS ---

            # Panel 1: Linke Spalte (Heatmaps)
            heatmap_x = 10

            # --- Kooperationsrate-Heatmap ---
            coop_heatmap_y = 10

            self._draw_horizontal_colorbar(screen, plt.get_cmap('RdYlGn'), heatmap_x, coop_heatmap_y, heatmap_width,
                                           colorbar_height, "Kooperationsrate", "0%", "100%")
            self._draw_heatmap(screen, grid_to_draw, heatmap_cell_size, heatmap_x,
                               coop_heatmap_y + title_space + label_space, 'cooperation_rate', 'RdYlGn')

            # --- Reward-Heatmap ---
            reward_heatmap_y = coop_heatmap_y + heatmap_height + spacing_between_heatmaps + title_space + label_space

            min_r, max_r = self._draw_heatmap(screen, grid_to_draw, heatmap_cell_size, heatmap_x,
                                              reward_heatmap_y + title_space + label_space, 'total_reward_mean', 'RdYlGn')
            self._draw_horizontal_colorbar(screen, plt.get_cmap('RdYlGn'), heatmap_x, reward_heatmap_y, heatmap_width,
                                           colorbar_height, "Total Reward Mean", f"{min_r:.0f}", f"{max_r:.0f}")

            # --- Kooperations-Payoff-Index Heatmap (links unten) ---
            payoff_heatmap_y = reward_heatmap_y + heatmap_height + spacing_between_heatmaps + title_space + label_space
            min_p, max_p = self._draw_heatmap(screen, grid_to_draw, heatmap_cell_size, heatmap_x,
                                              payoff_heatmap_y + title_space + label_space, 'strategic_cooperation_advantage', 'coolwarm_r')
            # Verwendung einer divergierende Colormap von Rot nach Blau
            self._draw_horizontal_colorbar(screen, plt.get_cmap('coolwarm_r'), heatmap_x, payoff_heatmap_y,
                                           heatmap_width, colorbar_height, "Strateg. Koop.-Vorteil", f"{min_p:.1f}",
                                           f"{max_p:.1f}")


            # Panel 2: Gitter der Strategietypen (Mitte)
            grid_x_start = left_panel_width
            for y in range(grid_shape[0]):
                for x in range(grid_shape[1]):
                    agent = grid_to_draw[y, x]
                    if agent is None: continue
                    color = get_agent_color_spectrum(agent)
                    rect = pygame.Rect(grid_x_start + x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (80, 80, 80), rect, 1)
                    # Agenten-ID zeichnen
                    id_number = str(agent.numeric_id)
                    brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                    text_surface = id_font.render(id_number, True, text_color)
                    text_rect = text_surface.get_rect(center=rect.center)
                    screen.blit(text_surface, text_rect)

            # Panel 3: Strategie-Liste (Rechts)
            panel_x_start = left_panel_width + grid_panel_width
            self._draw_strategy_panel(screen, font, grid_to_draw, panel_x_start, scroll_offset)


            # Info-Leiste (Unten)
            step_text = f"Match: {current_step}/{len(self.replay_history) - 1}"
            player_text = "| Startzustand"
            if current_step > 0 and active_players[0] is not None:
                player_text = f"| Aktuelles Duell: {active_players[0]} vs. {active_players[1]}"
            info_text = f"{step_text} {player_text}"
            text_surface = font.render(info_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, screen_height - 35))

            pygame.display.flip()
            clock.tick(30)

        pygame.key.set_repeat(0)
        pygame.quit()

    def _draw_strategy_panel(self, screen, font, grid, panel_x_start, scroll_offset):
        """
        Private Hilfsfunktion, um das scrollbare Strategie-Panel zu zeichnen.
        """
        title_font = pygame.font.Font(None, 24)
        agent_font = pygame.font.Font(None, 20)
        panel_rect = pygame.Rect(panel_x_start, 0, screen.get_width() - panel_x_start, screen.get_height())

        # Zeichne den Titel (fixe Position)
        title_surface = title_font.render("Agenten-Strategien (π)", True, (255, 255, 255))
        screen.blit(title_surface, (panel_x_start + 10, 10))

        y_start = 40
        y_offset = y_start - scroll_offset  # Wende den Scroll-Offset an
        line_height = 20

        # Iteriere durch alle Agenten im Gitter, um sie aufzulisten
        flat_agent_list = grid.flatten()

        for agent in flat_agent_list:
            # Zeichne nur die Zeilen, die im sichtbaren Bereich sind
            if y_offset > panel_rect.height:
                break  # Wir sind unterhalb des sichtbaren Bereichs
            if y_offset + line_height > y_start:  # Beginne erst im sichtbaren Bereich zu zeichnen
                # Agenten-ID und Farbe
                color = get_agent_color_spectrum(agent)
                id_surface = agent_font.render(agent.id, True, color)
                screen.blit(id_surface, (panel_x_start + 10, y_offset))

                # Strategie-Vektor
                policy = agent.get_policy()
                if policy.ndim == 2:
                    coop_policy = policy[:, 0]
                else:
                    coop_policy = policy

                strat_text = f"({coop_policy[0]:.2f}, {coop_policy[1]:.2f}, {coop_policy[2]:.2f}, {coop_policy[3]:.2f})"
                strat_surface = agent_font.render(strat_text, True, (200, 200, 200))
                screen.blit(strat_surface, (panel_x_start + 110, y_offset))

            y_offset += line_height

        # Optional: Zeichne eine Scrollbar-Andeutung
        total_content_height = len(flat_agent_list) * line_height
        if total_content_height > panel_rect.height - y_start:
            scrollbar_height = (panel_rect.height - y_start) / total_content_height * (panel_rect.height - y_start)
            scrollbar_y = y_start + (scroll_offset / total_content_height) * (panel_rect.height - y_start)
            pygame.draw.rect(screen, (80, 80, 80), (screen.get_width() - 10, scrollbar_y, 8, scrollbar_height))

    def _draw_heatmap(self, screen, grid, cell_size, x_offset, y_offset, data_type: str, cmap_name: str, max_reward_possible: float = 0):
        """
        Zeichnet eine Heatmap für einen gegebenen Datentyp ('cooperation_rate' oder 'total_reward').
        Gibt die Min/Max-Werte für die Legende zurück.
        """
        cmap = plt.get_cmap(cmap_name)
        grid_shape = grid.shape

        # Baue den korrekten Methodennamen zusammen (z.B. 'get_cooperation_rate')
        method_name = f'get_{data_type}'

        # Sammle alle relevanten Datenpunkte aus dem Gitter
        agent_data = [getattr(agent, method_name)() for agent in grid.flatten() if agent]
        if not agent_data:
            return 0, 0

        # Bestimme Min/Max-Werte der aktuellen Daten
        min_val, max_val = min(agent_data), max(agent_data)

        # Logik für die Normalisierung
        if data_type == 'cooperation_rate':
            # Feste Skala von 0 bis 1
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        elif data_type == 'total_reward':
            # Feste Skala von 0 bis zum theoretischen Maximum
            norm = mcolors.Normalize(vmin=0.0, vmax=max_reward_possible)
        elif data_type == 'cooperation_payoff_index':
            # Divergierende Skala mit Zentrum bei 0
            # Finde den größten Abstand von Null, um die Skala symmetrisch zu machen
            abs_max = max(abs(min_val), abs(max_val))
            if abs_max == 0: abs_max = 1 # Verhindere eine Skala von [-0, 0]
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)
        else: # Fallback für andere Daten
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

        for y in range(grid_shape[0]):
            for x in range(grid_shape[1]):
                agent = grid[y, x]
                if agent is None: continue

                value = getattr(agent, method_name)()

                # Normalisiere den Wert auf einen Bereich von 0.0 bis 1.0 für die Colormap
                normalized_val = norm(value)

                color_rgba = cmap(normalized_val)
                color_rgb = tuple(int(val * 255) for val in color_rgba[:3])

                rect = pygame.Rect(x_offset + x * cell_size, y_offset + y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, color_rgb, rect)

        return min_val, max_val


    def _draw_horizontal_colorbar(self, screen, cmap, x_pos, y_pos, width, height, title, min_val, max_val):
        """Zeichnet eine horizontale Farblegende."""
        font = pygame.font.Font(None, 18)
        title_font = pygame.font.Font(None, 20)

        title_surface = title_font.render(title, True, (255, 255, 255))
        screen.blit(title_surface, (x_pos, y_pos))

        bar_y = y_pos + 20
        for i in range(width):
            value = i / width
            color_rgba = cmap(value)
            color_rgb = tuple(int(val * 255) for val in color_rgba[:3])
            pygame.draw.line(screen, color_rgb, (x_pos + i, bar_y), (x_pos + i, bar_y + height))

        label_min = font.render(str(min_val), True, (255, 255, 255))
        label_max = font.render(str(max_val), True, (255, 255, 255))
        screen.blit(label_min, (x_pos, bar_y + height + 5))
        screen.blit(label_max, (x_pos + width - label_max.get_width(), bar_y + height + 5))