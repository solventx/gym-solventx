# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:46:30 2020

@author: splathottam
"""

from solventx import solventx as sx

config_file = "C:/Users/splathottam/Box Sync/GitHub/solventx/design.json"

sx_design = sx.solventx(config_file) #instantiate solvent extraction object 

sx_design.get_process() #Get process based on config file
print ('Number of components:{sx_design.num_input}')

sx_design.create_var_space(input_feeds=1) #Create variable space parameters
print(f'Combined variable space:{sx_design.combined_var_space}')

sx_design.evaluate_open(sx_design.design_variables) # evaluate design  using open loop recycle calculation - less accurate
print(f'Modules:{sx_design.modules}') 
print(f'Recovery:{sx_design.recovery}') 
print(f'Purity:{sx_design.purity}') 
print(f'Target_rees:{sx_design.target_rees}') 
print(f'Streams:{sx_design.streams}') 
