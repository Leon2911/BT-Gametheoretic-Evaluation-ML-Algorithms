from Main.Agenten.BaseAgent import BaseAgent
from Main.Agenten.Enums.PureStrategy import PureStrategy
from Main.IGD_Setup.Action import Action


class PureAgent(BaseAgent):
    def __init__(self, strategy_type=PureStrategy.ALWAYSCOOPERATE):
        super().__init__()
        self.type = strategy_type

    def choose_action(self, observation=None):
        if self.type == PureStrategy.ALWAYSCOOPERATE:
            # Always Cooperates
            return Action.COOPERATE

        elif self.type == PureStrategy.ALWAYSDEFECT:
            # Always defects
            return Action.DEFECT

        elif self.type == PureStrategy.TITFORTAT:
            # Cooperates in round 1, then mirrors last opponent action
            _, last_opponent_move = observation
            if last_opponent_move is None:
                return Action.COOPERATE
            return last_opponent_move

        elif self.type == PureStrategy.GRIMTRIGGER:
            # Starts cooperatively, defects permanently after first opponent defect
            if any(opp_move == Action.DEFECT for _, opp_move in observation):
                return Action.DEFECT
            return Action.COOPERATE

        elif self.type == "Random":
            import random
            return random.choice([Action.COOPERATE, Action.DEFECT])

        else:
            raise ValueError(f"Unknown strategy: {self.type}")

    def get_policy(self):
        if self.type == PureStrategy.ALWAYSCOOPERATE:
            return PureStrategy.ALWAYSCOOPERATE

        elif self.type == PureStrategy.ALWAYSDEFECT:
            return PureStrategy.ALWAYSDEFECT

        elif self.type == PureStrategy.TITFORTAT:
            return PureStrategy.TITFORTAT

        elif self.type == PureStrategy.GRIMTRIGGER:
            return PureStrategy.GRIMTRIGGER

        elif self.type == "Random":
            return "Random"

        else:
            raise ValueError(f"Unbekannte Strategie: {self.type}")