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

config_file = "../environment_design_config.json"
env_name = "gym_solventx-v0"

gym_env = gym.make(env_name, config_file=config_file) #Since suite_gym.load(env_name) doesn't work
env = suite_gym.wrap_env(gym_env)

time_step = env.reset()
print('Time step Spec:')
print(env.time_step_spec())
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Reward Spec:')
print(env.time_step_spec().reward)
print('Action Spec:')
print(env.action_spec())
print('Time step:')
print(time_step)

#action = np.array(1, dtype=np.int32)

#next_time_step = env.step(action)
#print('Next time step:')
#print(next_time_step)

#tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
