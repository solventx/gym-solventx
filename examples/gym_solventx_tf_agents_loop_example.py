# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:56:20 2020

@author: splathottam
"""

import time
import random 
import gym
import gym_solventx 
import os
import numpy as np

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

config_file = "../environment_design_config.json"
env_name = "gym_solventx-v0"

gym_env = gym.make(env_name, config_file=config_file) #Since suite_gym.load(env_name) doesn't work
py_env = suite_gym.wrap_env(gym_env)
tf_env = tf_py_environment.TFPyEnvironment(py_env)

print('Time step Spec:')
print(tf_env.time_step_spec())
print('Observation Spec:')
print(tf_env.time_step_spec().observation)
print('Reward Spec:')
print(tf_env.time_step_spec().reward)
print('Action Spec:')
print(tf_env.action_spec())

random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),tf_env.action_spec())


total_return = 0.0
n_episodes = 10
for i in range(n_episodes):
    time_step = tf_env.reset()
    
    print(f'Initial Time step at episode {i+1}:\n{time_step}')
    episode_return = 0.0
    step = 0
    while not time_step.is_last():
        action_step = random_policy.action(time_step)
        time_step = tf_env.step(action_step.action) #action_step.action
        print(f'Step:{step}:Reward:{time_step.reward},Observation:{time_step.observation}')
        
        episode_return += time_step.reward
        step = step + 1
    
    recovery = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.recovery.items() if key.startswith("Strip")}
    purity = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.purity.items() if key.startswith("Strip")}

    print(f'Recovery at episode {i+1}:{recovery}')
    print(f'Purity at episode {i+1}:{purity}')
    print(f'Design success at episode {i+1}:{tf_env._env._envs[0]._gym_env.env.design_success}')
    print(f'Total return at episode {i+1}:{episode_return}')
    total_return += episode_return

print(f'Average return after {n_episodes}:{total_return/n_episodes}')    
tf_env._env._envs[0]._gym_env.env.show_metric_statistics()
