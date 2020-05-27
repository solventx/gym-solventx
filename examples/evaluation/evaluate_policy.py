# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:17:22 2020

@author: splathottam
"""

import os
import numpy as np

import gym
import gym_solventx 

from gym_solventx.envs import utilities

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

import tensorflow as tf

np.set_printoptions(precision=3)


def evaluate_agent(policy_location,env_name,config_file,num_episodes = 1):
  """Test tf-agest policy"""
  
  tf_env = utilities.get_tf_env(env_name,config_file)
  policy = get_policy(policy_location)

  print(f'Policy type:{type(policy)}!')  
  print(f'Testing for {num_episodes} episodes!')
  
  returns = []
  recovery_list = []
  purity_list = []
  for episode in range(num_episodes):
    time_step = tf_env.reset()
    policy_state = policy.get_initial_state(tf_env.batch_size)

    step = 0
    print(f'Initial Time step:\n{time_step}')
    print(f'Initial policy state:{policy_state}!')
    episode_return = 0.0
    
    while not time_step.is_last():
      action_step = policy.action(time_step,policy_state)
      time_step = tf_env.step(action_step.action)
      print(f'Step:{step}:Reward:{time_step.reward},Observation:{time_step.observation}')
      #print(f'{step}:Policy state:{action_step.state}')
      
      episode_return += time_step.reward
      step = step + 1
      
    print(f'Total return at episode {episode+1}:{episode_return}')
    returns.append(episode_return.numpy())
    
    recovery = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.recovery.items() if key.startswith("Strip")}
    purity = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.purity.items() if key.startswith("Strip")}

    print(f'Recovery at episode {episode+1}:{recovery}')
    print(f'Purity at episode {episode+1}:{purity}')
    print(f'Design success at episode {episode+1}:{tf_env._env._envs[0]._gym_env.env.design_success}')
    print(f'Total return at episode {episode+1}:{episode_return}')
    
    recovery_list.append(recovery)
    purity_list.append(purity)
    
  print(f'List of returns after {num_episodes} episodes:{returns}')
  print(f'Average return:{np.mean(returns):.3f},Standard deviation:{np.std(returns):.3f}')
  tf_env._env._envs[0]._gym_env.env.show_all_initial_metrics()
  tf_env._env._envs[0]._gym_env.env.show_all_final_metrics()
  tf_env._env._envs[0]._gym_env.env.show_initial_metric_statistics()
  tf_env._env._envs[0]._gym_env.env.show_final_metric_statistics()
  tf_env._env._envs[0]._gym_env.env.show_initial_design()
  tf_env._env._envs[0]._gym_env.env.show_final_design()
  print(f'Solvent extraction state:{tf_env._env._envs[0]._gym_env.env.sx_design.x}')
  tf_env._env._envs[0]._gym_env.env.save_metrics()
  tf_env._env._envs[0]._gym_env.env.save_design()
  
def get_policy(policy_location):
  """Load policy"""
  
  print(f'Loading policy at {policy_location}')
  return tf.compat.v2.saved_model.load(policy_location)

 

def main():
    policy_dir = r"C:\Users\splathottam\Box Sync\GitHub\gym-solventx\examples\ppo_rnn_rev4\tensorboard\gym_solventx-v0\policy_saved_model"
    policy_file = r"policy_000071300"
    env_name = 'gym_solventx-v0'
    config_file = r"C:\Users\splathottam\Box Sync\GitHub\gym-solventx\environment_design_config.json"
    num_episodes = 50
    
    evaluate_agent(os.path.join(policy_dir,policy_file),env_name,config_file,num_episodes)

if __name__ == '__main__':
    main()