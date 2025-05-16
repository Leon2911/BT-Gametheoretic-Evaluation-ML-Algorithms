from Main.IGD_Setup.Action import Action


class PPOAgentWrapper:
    def __init__(self, model, env):
        self.last_action = None
        self.model = model
        self.env = env
        self.last_observation, _ = env.reset()

    def choose_action(self, opponent=None):
        action, _states = self.model.predict(self.last_observation, deterministic=True)
        self.last_action = action
        self.last_observation, _, _, _, _ = self.env.step(action)
        return Action.COOPERATE if action == 0 else Action.DEFECT

    def receive_reward(self, reward):
        pass

    def update_memory(self, my_move, opponent_move):
        pass

    @property
    def id(self):
        return "PPO_Agent"
