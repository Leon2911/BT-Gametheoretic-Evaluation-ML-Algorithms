from abc import ABC

from Main.IGD_Setup.Action import Action


def format_strategy_vector(strategy_matrix):
    """
    Konvertiert eine Strategie (egal ob 4x2 Matrix oder 4er Vektor)
    in einen lesbaren String, der die Kooperationswahrscheinlichkeit anzeigt.
    """

    # Prüfe, ob die übergebene Policy eine 2D-Matrix ist (von QL/SARSA)
    if strategy_matrix.ndim == 2:
        # Extrahiere die erste Spalte (Kooperations-Wahrscheinlichkeiten)
        coop_vector = strategy_matrix[:, 0]
    else:
        # Die Policy ist bereits der korrekte 1D-Vektor (von PureAgent)
        coop_vector = strategy_matrix

    p_c_given_cc = coop_vector[0]
    p_c_given_cd = coop_vector[1]
    p_c_given_dc = coop_vector[2]
    p_c_given_dd = coop_vector[3]

    # Format the output string with 2 decimal places
    vector_string = (
        f"π = ({p_c_given_cc:.2f}, "
        f"{p_c_given_cd:.2f}, "
        f"{p_c_given_dc:.2f}, "
        f"{p_c_given_dd:.2f})"
    )

    return vector_string

class BaseAgent(ABC):
    next_id = 0

    def __init__(self, name=None, initial_action=Action.COOPERATE, model=None):
        self.id = f"Agent{BaseAgent.next_id}"
        BaseAgent.next_id += 1
        self.name = name if name is not None else self.id

        # Rewards
        self.total_reward = 0

        # Initial start action
        self.initial_action = initial_action

        # Counters for statistics
        self.action_count = 0
        self.cooperation_count = 0

        # Optional: RL model (e.g. PPO, DQN, etc. from SB3)
        self.model = model


    def receive_reward(self, reward):
        """Accumulates the reward received by the agent over its lifecycle"""
        self.total_reward += reward

    def get_total_reward(self):
        """Return the accumulated reward of the agent"""
        return self.total_reward

    def reset_stats(self):
        """Resets all statistical counters (rewards, actions, etc.)"""
        self.total_reward = 0
        self.action_count = 0
        self.cooperation_count = 0

    def log_action(self, action):
        """Protokolliert eine ausgeführte Aktion für die Statistik."""
        self.action_count += 1
        if action == Action.COOPERATE:
            self.cooperation_count += 1

    def get_cooperation_rate(self):
        """Returns the cooperation rate as a value between 0 and 1"""
        if self.action_count == 0:
            return 0.0  # Prevents division by zero
        return self.cooperation_count / self.action_count

