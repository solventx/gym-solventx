# -*- coding: utf-8 -*-
"""reinforce_agent_simulator_test.py"""

import gym
import gym_solventx
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt

#import tensorflow_probability as tfp
from tf_agents.utils            import common
from tf_agents.environments     import suite_gym
from tf_agents.metrics          import tf_metrics
from tf_agents.trajectories     import trajectory
from tf_agents.eval             import metric_utils
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments     import tf_py_environment
from tf_agents.drivers          import dynamic_step_driver
from datetime                   import datetime as datetime
from tf_agents.replay_buffers   import tf_uniform_replay_buffer
from tf_agents.networks         import actor_distribution_network

tf.compat.v1.enable_v2_behavior()

print(tf.__version__)
#print(tfp.__version__)

env_name = 'gym_solventx-v0'         # @param
num_iterations = 20                   # @param
collect_episodes_per_iteration = 2    # @param
replay_buffer_capacity         = 2000 # @param

fc_layer_params = (100,50,25)

log_data = pd.DataFrame(columns=['Action', 'Observation', 'Reward']) #Dictionary holding log data as dataframe

log_interval      = 2     # @param
eval_interval     = 1     # @param
num_eval_episodes = 1     # @param
learning_rate     = 1e-3  # @param
max_steps         = 500

eval_py_env  = suite_gym.load(env_name, max_episode_steps=1)
train_py_env = suite_gym.load(env_name, max_episode_steps=max_steps)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env  = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy    = tf_agent.policy
collect_policy = tf_agent.collect_policy

def Log_Results(file, num_episodes=2):
  for _ in range(num_episodes):
      time_step = eval_env.reset()
      while not time_step.is_last():
          action_step = tf_agent.policy.action(time_step)
          time_step   = eval_env.step(action_step.action)
          output      = eval_py_env.render(mode='file')
          file.write(output)
      file.write(''.join(["=" for _ in range(100)]) + '\n')

try:

  #@test {"skip": true}
  def compute_avg_return(environment, policy, step, num_episodes=10):

    total_return = 0.0
  #    for _ in range(num_episodes):
    for _ in range(1):

      episode_return = 0.0
      time_step      = environment.reset()

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step   = environment.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return

  #    avg_return = total_return / num_episodes
    avg_return = total_return
    return avg_return.numpy()[0]

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec  = tf_agent.collect_data_spec,
      batch_size = train_env.batch_size,
      max_length = replay_buffer_capacity)\

  def collect_episode(environment, policy, num_episodes):
      global log_data
      
      episode_counter = 0
      environment.reset()
      
      print('Resetting...')
      while episode_counter < num_episodes:
          time_step      = environment.current_time_step()
          action_step    = policy.action(time_step)
          next_time_step = environment.step(action_step.action)
          traj           = trajectory.from_transition(time_step, action_step, next_time_step)

          #Capture Training Decisions
          action      = action_step.action
          observation = f'{next_time_step.observation}'
          reward      = next_time_step.reward
          newData     = {'Action':action,'Observation':observation, 'Reward':reward}
          log_data    = log_data.append(pd.DataFrame(newData), ignore_index=True)

          # Add trajectory to the replay buffer
          replay_buffer.add_batch(traj)
          
          if traj.is_boundary():
              episode_counter += 1

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  tf_agent.train = common.function(tf_agent.train)

  # Reset the train step
  tf_agent.train_step_counter.assign(0)

  # Evaluate the agent's policy once before training.
  avg_return = compute_avg_return(eval_env, tf_agent.policy, 1, num_eval_episodes)
  returns    = [avg_return]


  for _ in range(num_iterations):
      
      # Collect a few episodes using collect_policy and save to the replay buffer.
      collect_episode(
        train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

      # Use data from the buffer and update the agent's network.
      experience = replay_buffer.gather_all()
      train_loss = tf_agent.train(experience)
      replay_buffer.clear()
      
      step = tf_agent.train_step_counter.numpy()

      if step % log_interval == 0:
          print('step = {0}: loss = {1}'.format(step, train_loss.loss))

      if step % eval_interval == 0:
          avg_return = compute_avg_return(eval_env, tf_agent.policy, step, num_eval_episodes)
          print('step = {0}: Average Return = {1}'.format(step, avg_return))
          returns.append(avg_return)

  steps = range(0, num_iterations + 1, eval_interval)
  plt.plot(steps, returns)
  plt.ylabel('Average Return')
  plt.xlabel('Step')
  #plt.ylim(top=5)
        
  # #Log evaluation results
  # with open('output2.txt', 'w') as file:
  #   Log_Results(file)

  eval_py_env.render(mode='human')
except:
  print('An Exception Occurred!!!')
finally:
  with open (f'gym_solventx/envs/methods/output/train_log_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv', 'w+') as log_file:
    log_file.write(f'"Epochs: {num_iterations}", "Step Limit: {max_steps}"\n')
    log_file.write('Max Result: ' + str(log_data['Reward'].max()) + '\n')
    log_data.to_csv(log_file)
    print(log_data.sort_values(by='Reward', ascending=False).head(5))
