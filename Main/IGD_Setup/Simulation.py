from typing import Dict, Tuple

from Main.Evaluation.Evaluation import Evaluation
from Main.IGD_Setup.Action import Action

class Simulation:
    DEFAULT_PAYOFF_MATRIX: Dict[Tuple[str, str], Tuple[int, int]] = {
        (Action.COOPERATE, Action.COOPERATE): (3, 3),
        (Action.COOPERATE, Action.DEFECT): (0, 5),
        (Action.DEFECT, Action.COOPERATE): (5, 0),
        (Action.DEFECT, Action.DEFECT): (1, 1)
    }

    def __init__(self, agent1, agent2, total_rounds=100, payoff_matrix=None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.total_rounds = total_rounds
        self.payoff_matrix = payoff_matrix if payoff_matrix else self.DEFAULT_PAYOFF_MATRIX

    def play_ipd(self):
        for _ in range(self.total_rounds):
            self.play_ipd_round()


    def play_ipd_round(self):
        action1 = self.agent1.choose_action()
        action2 = self.agent2.choose_action()

        reward1, reward2 = self.calculate_reward(action1, action2)

        self.agent1.receive_reward(reward1)
        self.agent2.receive_reward(reward2)
        self.agent1.update_memory(action1, action2)
        self.agent2.update_memory(action2, action1)

        ## Daten erfassen
        #Evaluation.capture_vector(self.agent1.id, self.agent1.strategy_vector)
        #Evaluation.capture_vector(self.agent2.id, self.agent2.strategy_vector)
        #Evaluation.capture_reward(self.agent1.id, reward1)
        #Evaluation.capture_reward(self.agent2.id, reward2)
        #Evaluation.capture_action(self.agent1.id, action1, self.agent2.id, action2)

    def calculate_reward(self, a1, a2):
        return self.payoff_matrix.get((a1, a2), (0, 0))