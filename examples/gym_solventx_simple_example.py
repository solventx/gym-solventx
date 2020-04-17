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

print(f'Observation:{observation}\nReward:{env.reward},')