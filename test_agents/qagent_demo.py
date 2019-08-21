import gym
import gym_solventx
from qagent import QAgent

env = gym.make('gym_solventx-v0', goals_list=['Purity', 'Recovery'])

qagent = QAgent(env)
qagent.train_agent()
qagent.show_plots()