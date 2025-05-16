from Main.Agenten.ML_Wrapper import PPOAgentWrapper
from Main.Agenten.PureAgent import PureAgent
from Main.Evaluation.Evaluation import Evaluation
from Main.IGD_Setup.Simulation import Simulation
from stable_baselines3 import PPO

from Main.IGD_Setup.IPDEnv import IPDEnv

if __name__ == '__main__':

        evaluation = Evaluation()
        env = IPDEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5000)

        agent1 = PureAgent(strategy_type="TitForTat")
        agent2 = PPOAgentWrapper(model, env)

        IPD = Simulation(agent1, agent2)
        IPD.play_ipd()
        # Ergebnisse analysieren
        #evaluation.plot_all()







        # init agents
        # init ipd
        # init Gamefield (later)

        # while(True):
        #    if(two agents meet or get chosen to interact):
        #       play_ipd(agent1, agent2, total_rounds)
        #       optimize_vector
        #       capture_globals #Variables like total_reward, strategy_vector, etc. in appropriate data structure
        #    if(user stops simulation):
        #       evaluate_captured_information

        # agent1 = PureAgent(strategy_type="AlwaysDefect")
        # agent1 = PPOAgent(training_iters=2000)
        # agent2 = PureAgent(strategy_type="TitForTat")

        # IPD = PrisonersDilemmaGame(agent1, agent2)
        # IPD.play_ipd()
        # print("PPO Agent Reward:", agent1.total_reward)

        # Erstelle PPO-Agent und Gegner
        # ppo_agent = PPOAgent(memory_length=1, hidden_dim=64, training_iters=10)
        ##opponent = PureAgent(strategy_type="TitForTat")
        # opponent = PureAgent(strategy_type="AlwaysDefect")
        #
        #
        ## Trainiere PPO-Agent gegen den Gegner
        # print("Training PPO-Agent...")
        # ppo_agent.optimize(opponent, episodes=100, steps_per_episode=50)
        #
        ## Zeige resultierende Strategie
        # print("PPO-Agent Strategie nach Training:")
        # print(f"PCC: {ppo_agent.strategy_vector[0]:.2f}")
        # print(f"PCD: {ppo_agent.strategy_vector[1]:.2f}")
        # print(f"PDC: {ppo_agent.strategy_vector[2]:.2f}")
        # print(f"PDD: {ppo_agent.strategy_vector[3]:.2f}")
        #
        ## Teste den trainierten Agenten in einem vollständigen Spiel
        # print("\nTesten des trainierten Agenten gegen AlwaysDefect:")
        # game = PrisonersDilemmaGame(ppo_agent, opponent, total_rounds=300)
        # game.play_ipd()
        #
        # print(f"PPO-Agent Belohnung: {ppo_agent.total_reward}")
        # print(f"Gegner Belohnung: {opponent.total_reward}")

        # Simulation einrichten