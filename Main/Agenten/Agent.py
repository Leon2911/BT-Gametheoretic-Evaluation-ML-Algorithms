# Base Agent Class
from collections import deque
import numpy as np
import random

from Main.IGD_Setup.Action import Action


class Agent:
    next_id = 0

    def __init__(self, memory_length=1, initial_strategy=None):
        # ID-Zuweisung
        self.total_reward = 0
        self.id = f"Agent{Agent.next_id}"
        Agent.next_id += 1

        # Strategie als Vektor
        if initial_strategy is None:
            self.strategy_vector = [0.5, 0.5, 0.5, 0.5]  # [PCC, PCD, PDC, PDD]
        else:
            self.strategy_vector = initial_strategy

        # Gedächtnis als Deque
        self.memory = deque(maxlen=memory_length)

    def get_last_interaction(self):
        """Holt die letzte Interaktion aus dem Gedächtnis"""
        if not self.memory:
            return None, None
        return self.memory[-1]

    def update_memory(self, my_move, opponent_move):
        """Aktualisiert das Gedächtnis nach einer Spielrunde"""
        self.memory.append((my_move, opponent_move))

    def get_strategy_index(self, history=None):
        """
        Bestimmt den Index im Strategievektor basierend auf der Geschichte
        Standard: Betrachtet nur die letzte Interaktion (für PCC, PCD, PDC, PDD)
        """
        if history is None:
            my_last_move, opponent_last_move = self.get_last_interaction()
            if my_last_move is None or opponent_last_move is None:
                return None  # Keine Historie verfügbar

            # Standard-Indizierung für 1-Schritt-Historie
            idx = 0
            if my_last_move:  # True = Cooperation
                if opponent_last_move:  # True = Cooperation
                    idx = 0  # CC
                else:
                    idx = 1  # CD
            else:  # False = Defection
                if opponent_last_move:
                    idx = 2  # DC
                else:
                    idx = 3  # DD

            return idx
        else:
            # Hier kann man komplexere Indizierungen für mehrere Schritte implementieren
            # Zum Beispiel könnte man binäre Codierung verwenden:
            # CC CC DC -> 1100 -> 12
            # Zum Beispiel falls history==3 ist oder sowas
            pass

    def choose_action(self, opponent=None):
        """Wählt die nächste Aktion basierend auf dem Gedächtnis"""
        idx = self.get_strategy_index()

        # Erste Runde oder kein Index ermittelbar
        if idx is None:
            return random.choice([Action.COOPERATE, Action.DEFECT])

        prob = self.strategy_vector[idx]
        return Action.COOPERATE if random.random() < prob else Action.DEFECT

    def receive_reward(self, reward):
        """Kann genutzt werden, um Belohnung zu akkumulieren oder zu analysieren"""
        if not hasattr(self, "total_reward"):
            self.total_reward = None
        self.total_reward += reward

    def optimize(self, opponent_history, score):
        """Optimiere Agenten-Strategie basierend auf letzte Spielrunde"""
        # Hier würde der ML-Algorithmus implementiert werden vermutlich
        pass