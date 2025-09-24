from enum import Enum

# Model identification names
class Model(Enum):
    # Tabellarisch
    Q_LEARNING = "Q_LEARNING"
    WoLF_PHC = "WoLF_PHC"
    SARSA = "SARSA"

    # SB3 Models
    PPO = "PPO"
    DQN = "DQN"
    REINFORCE = "REINFORCE"