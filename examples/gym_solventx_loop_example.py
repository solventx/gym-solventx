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

config_file = "../environment_design_config.json"

env = gym.make('gym_solventx-v0', config_file=config_file,identifier='test')
observation = env.reset()
print(f'Initial Observation:{observation}\nReward:{env.reward},')

for step in range(1,10):
    action = env.action_space.sample() 
    print(f'Step:{step},Action:{action}')        
    observation, reward, done,_ = env.step(action)
    print(f'Observation:{observation}\nReward:{reward}\nDone:{done}')
    env.decipher_action(action)

print(f'Final Observation:{observation}\nReward:{env.reward},')
    

    