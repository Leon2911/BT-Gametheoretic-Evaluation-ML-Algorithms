import random
import numpy as np
from Main.Agenten.BaseAgent import BaseAgent
from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.IGD_Setup.Action import Action


class PureAgent(BaseAgent):
    def __init__(self, strategy_type: PureStrategy = PureStrategy.TITFORTAT):
        """
        Initialisiert einen Agenten mit einer festen, deterministischen Strategie.

        Args:
            strategy_type (PureStrategy): Das Enum-Mitglied der Strategie.
        """
        super().__init__()
        self.strategy_type = strategy_type
        self.id = f"Pure_{self.strategy_type.value}_{self.id}"  # Eindeutige ID

        # Interner Zustand, der für manche Strategien benötigt wird
        self.has_been_betrayed = False

    def reset_stats(self):
        """Setzt interne Zustände und Statistiken für ein neues Match zurück."""
        super().reset_stats()
        self.has_been_betrayed = False

    def choose_action(self, observation: int) -> Action:
        """
        Wählt eine Aktion basierend auf der festen Strategie und der Observation.
        Observation ist ein Integer (0-3), der das Ergebnis der letzten Runde darstellt:
        0=(C,C), 1=(C,D), 2=(D,C), 3=(D,D) aus der "Ich"-Perspektive.

        Returns:
            Action: Das Enum der gewählten Aktion (COOPERATE oder DEFECT).
        """

        # --- STRATEGIE-LOGIK BASIEREND AUF INTEGER-OBSERVATION ---

        if self.strategy_type == PureStrategy.ALWAYSCOOPERATE:
            return Action.COOPERATE

        elif self.strategy_type == PureStrategy.ALWAYSDEFECT:
            return Action.DEFECT

        elif self.strategy_type == PureStrategy.TITFORTAT:
            # "Spiegle den letzten Zug des Gegners."
            # Der Gegner hat kooperiert in Zustand 0 (CC) und 2 (DC).
            if observation in [0, 2]:
                return Action.COOPERATE
            else:  # Der Gegner hat defektiert in Zustand 1 (CD) und 3 (DD).
                return Action.DEFECT

        elif self.strategy_type == PureStrategy.GRIMTRIGGER:
            # "Kooperiere, bis du einmal betrogen wirst, dann defektiere für immer."
            if self.has_been_betrayed:
                return Action.DEFECT

            # Prüfe, ob der Gegner in der letzten Runde defektiert hat.
            # Gegner hat defektiert in Zustand 1 (CD) und 3 (DD).
            if observation in [1, 3]:
                self.has_been_betrayed = True  # Merke dir den Verrat!
                return Action.DEFECT
            else:
                return Action.COOPERATE

        elif self.strategy_type == "Random":  # Behalte String für Flexibilität oder ändere zu Enum
            return Action(random.choice([0, 1]))

        else:
            raise ValueError(f"Unbekannte Strategie im PureAgent: {self.strategy_type}")

    def get_policy(self) -> np.ndarray:
        # Diese Methode ist bereits mit der neuen Logik kompatibel,
        # da sie die Policy als Vektor von Wahrscheinlichkeiten zurückgibt.
        if self.strategy_type == PureStrategy.ALWAYSCOOPERATE:
            return np.array([1.0, 1.0, 1.0, 1.0])
        elif self.strategy_type == PureStrategy.ALWAYSDEFECT:
            return np.array([0.0, 0.0, 0.0, 0.0])
        elif self.strategy_type == PureStrategy.TITFORTAT:
            return np.array([1.0, 0.0, 1.0, 0.0])
        elif self.strategy_type == PureStrategy.GRIMTRIGGER:
            if self.has_been_betrayed:
                return np.array([0.0, 0.0, 0.0, 0.0])
            else:
                return np.array([1.0, 0.0, 1.0, 0.0])
        elif self.strategy_type == "Random":
            return np.array([0.5, 0.5, 0.5, 0.5])
        else:
            raise ValueError(f"Unbekannte Strategie: {self.strategy_type}")

    def train(self, experiences):
        """Ein PureAgent lernt nicht. Diese Methode tut nichts."""
        pass