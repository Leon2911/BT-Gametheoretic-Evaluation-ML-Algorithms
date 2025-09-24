from stable_baselines3 import DQN

from Main.Agenten.BaseAgent import BaseAgent
from Main.Agenten.Enums.RL_Model import Model
from Main.IGD_Setup.IPDEnv import IPDEnv


class SB3Agent(BaseAgent):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)

        env = IPDEnv()
        # TODO Doesnt work because SB3 only works with gymnasium envs not pettingzoo envs, supersuit needed

        # Configure SB3 model if specified
        if model == Model.DQN:
            self.model = DQN("MlpPolicy", env, verbose=0)
        else:
            raise ValueError(f"SB3 model {model} not supported.")

    def choose_action(self, observation=None):
        if observation is None:
            # Default start observation (0 = CC, both cooperated)
            observation = 0
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)