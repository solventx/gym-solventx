# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:17:22 2020

@author: splathottam
"""

import os
import numpy as np

from gym_solventx.envs import env_utilities,evaluation_utilities
from solventx import utilities as util

np.set_printoptions(precision=3)

def main():
    process_config_file = r"/home/splathottam/GitHub/solventx/configurations/design_c.json"       
    variable_config_file = r"/home/splathottam/GitHub/solventx/env_design_config.json"
    
    n_cases = 20
    
    print(f'Generating {n_cases} cases for feed input...')
    cases = util.generate(util.read_config(process_config_file),n_cases)
      
    optimization_results_df = evaluation_utilities.optimization_evaluation_loop(process_config_file,variable_config_file,cases)

    

if __name__ == '__main__':
    main()