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

config_file = "..\\environment_design_config.json"

env = gym.make('gym_solventx-v0', config_file=config_file)
observation = env.reset()
print(f'Initial Observation:{observation}\nInitial Reward:{env.reward},')

print(f'Executing one step:')
action = env.action_space.sample() 
observation, reward, done,_ = env.step(action)

print(f'Action:{action}\nObservation:{observation}\nReward:{reward}\nDone:{done}')
env.decipher_action(action)
#print(env.action_dict)

    