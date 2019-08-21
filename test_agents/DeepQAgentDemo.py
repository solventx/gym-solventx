import gym, os
from os      import path
import gym_solventx
from test_agents.DQAgent import DQAgent

env = gym.make('gym_solventx-v0', goals_list=['Purity', 'Recovery'])

agent_opts = {
                #hyperparameters
                'BATCH_SIZE':             64,
                'EPSILON_START':         .85,
                'EPSILON_DECAY':         .99,
                'DISCOUNT':              .99,
                'MAX_STEPS':             500,
                'MIN_EPSILON' :          0.05,
                'REPLAY_MEMORY_SIZE':    1000,
                'LEARNING_RATE':         0.0001,
                'ACTION_POLICY':         'eg',

                #saving and logging results
                'AGGREGATE_STATS_EVERY':   2,
                'SHOW_EVERY':              1,
                'COLLECT_RESULTS':      True,
                'COLLECT_CUMULATIVE':   False,
                'SAVE_EVERY_EPOCH':     True,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      './test_agents/best_results/sx_best_model',
            } 

model_opts = {
                'num_layers':      4,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'add_dropout':     False,
                'add_callbacks':   False,
                'activation':      'linear',
                'nodes_per_layer': [64,64,32,32],
            }

# Train models
def train_model(agent_opts, model_opts):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    agent.train(n_epochs=75, render=False)
    agent.save_weights('./test_agents/best_results/sx_best_model')
    agent.show_plots()
    agent.show_plots('loss')
    env.close()

#Evaluate model
def evaluate_model(agent_opts, model_opts, best_model=True):
    agent = DQAgent(env, **agent_opts)
    agent.build_model(**model_opts)
    if best_model:
      filename = agent_opts.get('BEST_MODEL_FILE')[:-3]
      agent.load_weights(filename)
    else:
      agent.load_weights('./test_agents/best_results/sx_best_model')
    agent.evaluate(1, render=True, verbose=False)


train_model(agent_opts, model_opts)
evaluate_model(agent_opts, model_opts, best_model=True)
