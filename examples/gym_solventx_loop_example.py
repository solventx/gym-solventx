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

config_file = "C:\\Users\\splathottam\\Box Sync\\GitHub\\gym-solventx\\environment_design_config.json"

env = gym.make('gym_solventx-v0', config_file=config_file)
observation = env.reset()
print(f'Initial Observation:{observation}\nReward:{env.reward},')

for step in range(10):
    action = env.action_space.sample() 
    observation, reward, done,_ = env.step(action)
    print(f'Step:{step}')
    print(f'Action:{action}\nObservation:{observation}\nReward:{reward}\nDone:{done}')

print(f'Final Observation:{observation}\nReward:{env.reward},')
    

    