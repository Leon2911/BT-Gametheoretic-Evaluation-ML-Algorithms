from enum import Enum

# Possible pure strategies in the Iterated Prisoners Dilemma
class PureStrategy(Enum):

    TITFORTAT = "TitForTat"
    ALWAYSDEFECT = "AlwaysDefect"
    ALWAYSCOOPERATE = "AlwaysCooperate"
    GRIMTRIGGER = "GrimTrigger"