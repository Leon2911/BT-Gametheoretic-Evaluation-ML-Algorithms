from Main.Agenten.Agent import Agent
from Main.IGD_Setup.Action import Action


class PureAgent(Agent):
    def __init__(self, strategy_type="AlwaysCooperate", memory_length=1):
        super().__init__(memory_length=memory_length, initial_strategy=None)
        self.type = strategy_type

    def choose_action(self, opponent=None):
        if self.type == "AlwaysCooperate":
            return Action.COOPERATE

        elif self.type == "AlwaysDefect":
            return Action.DEFECT

        elif self.type == "TitForTat":
            # Kooperiert in Runde 1, dann spiegelt letzte Gegneraktion
            _, last_opponent_move = self.get_last_interaction()
            if last_opponent_move is None:
                return Action.COOPERATE
            return last_opponent_move

        elif self.type == "GrimTrigger":
            # Startet kooperativ, defektiert für immer nach erstem Defekt
            if any(opp_move == Action.DEFECT for _, opp_move in self.memory):
                return Action.DEFECT
            return Action.COOPERATE

        elif self.type == "Random":
            import random
            return random.choice([Action.COOPERATE, Action.DEFECT])

        else:
            raise ValueError(f"Unbekannte Strategie: {self.type}")
