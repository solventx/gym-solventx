# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:46:30 2020

@author: splathottam
"""
#WOrking configurations: b,c,f
from solventx import solventx as sx
from solventx import utilities
import numpy as np

config_file = "C:/Users/splathottam/Box Sync/GitHub/solventx/design_c.json" #build configs
confDict = utilities.read_config(config_file)

config_env_file = "C:/Users/splathottam/Box Sync/GitHub/solventx/env_design_config.json"
confEnvDict = utilities.get_env_config_dict(config_env_file)

cases = utilities.generate(confDict,1) #generate case
print(cases)
ree_mass = [item for item in cases['0'].values()] #select case

sx_design = sx.solventx(confDict, confEnvDict, ree_mass) #instantiate solvent extraction object 

sx_design.get_process() #Get process based on config file
print (f'Number of components:{sx_design.num_input}')

#sx_design.create_var_space(input_feeds=1) #Create variable space parameters
print(f'Combined variable space:{sx_design.combined_var_space}')
print(f'Mod space:{sx_design.mod_space}')

print(f'Feed input:{sx_design.ree_mass}')
#sx_design.x = np.array([0.599971,0.0001,0.0001,2.0,4.136241,3.618035, 2.558954, 0.644779,6.0,6.0,4.0])
sx_design.evaluate(sx_design.design_variables) # evaluate design  using open loop recycle calculation - less accurate
print(f'x0:{sx_design.x}')
print(f'Modules:{sx_design.modules}') 
print(f'Recovery:{sx_design.recovery}') 
print(f'Purity:{sx_design.purity}') 
print(f'Target_rees:{sx_design.target_rees}') 
print(f'Streams:{sx_design.streams}') 
