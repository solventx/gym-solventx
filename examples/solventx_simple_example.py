# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:46:30 2020

@author: splathottam
"""
#WOrking configurations: b,c,f
from solventx import solventx as sx
import numpy as np

config_file = "C:/Users/splathottam/Box Sync/GitHub/solventx/updated_design.json"

sx_design = sx.solventx(config_file) #instantiate solvent extraction object 

sx_design.get_process() #Get process based on config file
print (f'Number of components:{sx_design.num_input}')

sx_design.create_var_space(input_feeds=1) #Create variable space parameters
print(f'Combined variable space:{sx_design.combined_var_space}')
print(f'Feed input:{sx_design.ree_mass}')
sx_design.x = np.array([0.599971,0.0001,0.0001,2.0,4.136241,3.618035, 2.558954, 0.644779,6.0,6.0,4.0])
sx_design.evaluate_open(sx_design.x) # evaluate design  using open loop recycle calculation - less accurate
print(f'x0:{sx_design.x}')
print(f'Modules:{sx_design.modules}') 
print(f'Recovery:{sx_design.recovery}') 
print(f'Purity:{sx_design.purity}') 
print(f'Target_rees:{sx_design.target_rees}') 
print(f'Streams:{sx_design.streams}') 
