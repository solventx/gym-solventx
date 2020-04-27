# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:17:22 2020

@author: splathottam
"""
import numpy as np

import gym
import gym_solventx 

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

import tensorflow as tf

np.set_printoptions(precision=3)
policy_dir = "/home/splathottam/tmp/dqn/gym/gym_solventx-v0/policy_saved_model/policy_000050000"
env_name='gym_solventx-v0'
config_file = "/home/splathottam/GitHub/gym-solventx/environment_design_config.json"

def test_agent(policy, eval_tf_env,num_episodes = 10):
  """Test tf-agest policy"""

  print(f'Found policy of type:{type(saved_policy)}!')
  print(f'Testing for {num_episodes} episodes!')
  returns = []
  for episode in range(num_episodes):
    time_step = eval_tf_env.reset()
    print(f'Initial Time step:\n{time_step}')
    episode_return = 0.0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      print(f'Reward:{time_step.reward},Observation:{time_step.observation}')
      episode_return += time_step.reward
    
    print(f'Total return at episode {episode+1}:{episode_return}')
    returns.append(episode_return.numpy())
  print(f'List of returns after {num_episodes} episodes:{returns}')
  print(f'Average return:{np.mean(returns):.3f},Standard deviation:{np.std(returns):.3f}')
      

eval_gym_env = gym.make(env_name, config_file=config_file)
eval_py_env = suite_gym.wrap_env(eval_gym_env,max_episode_steps=100)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)    

saved_policy = tf.compat.v2.saved_model.load(policy_dir)

test_agent(saved_policy, eval_tf_env)
