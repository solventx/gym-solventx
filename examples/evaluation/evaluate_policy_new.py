# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:17:22 2020

@author: splathottam
"""

import os
import numpy as np

from gym_solventx.envs import env_utilities,evaluation_utilities

np.set_printoptions(precision=3)

def main():
    process_config_file = r"C:\Users\splathottam\Box Sync\GitHub\solventx\configurations\design_d.json"       
    variable_config_file = r"C:\Users\splathottam\Box Sync\GitHub\solventx\env_design_config.json"

    env_config_file = r"C:\Users\splathottam\Box Sync\GitHub\gym-solventx\environment_design_config.json"
    policy_dir = r"C:\Users\splathottam\Box Sync\GitHub\gym-solventx\examples\ppo_rnn_rev6\tensorboard\gym_solventx-v0\policy_saved_model"
    policy_file = r"policy_000000900" #r"policy_000228600"#r"policy_000204500" #r"policy_000124100" #r"policy_000204500" #r"policy_000132700"
    num_cases = 3
      
    evaluation_utilities.compare_agent_with_optimization(process_config_file,variable_config_file,env_config_file,policy_dir,policy_file,n_cases = num_cases)

if __name__ == '__main__':
    main()