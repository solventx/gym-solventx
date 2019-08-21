# -*- coding: utf-8 -*-
"""
Front Page

Created on Mon May 20 14:11:15 2019

@author: ciloeje
"""


""" imports """
import numpy as np
import pandas as pd
import plotlyon as pl
import gym_solventx.envs.methods.solvent_sweep as ss

#data = pd.read_csv('input/data.csv')
#bounds = pd.read_csv('input/bounds.csv')
#
#print('data', data[['OA Extraction']])
#print()
#print('bounds', bounds[['OA Extraction']].iloc[1][0])


""" Initialize Column Object """
obj = ss.solvent_extraction(scale=1, feedvol = .02) # liters per minute

#s = '''[[2.5000000e-01 1.7600644e-02 2.1113141e-05 1.5589986e+00 4.2999992e+00 6.0000002e-01 3.4500000e+00 6.0000002e-01 2.0000000e+00 7.0000000e+00 5.0000000e+00]]'''
import numpy as np
#obj.variables = np.array(list(map(float, s[2:-2].split(' '))))
#print(obj.variables)
obj.variables = np.array([3.00000000e-01, 6.90450988e-04, 1.28287483e-05, 1.00651110e-01,
       1.65000000e+00, 2.50000000e+00, 2.30000000e+00, 5.00000000e-02,
       8.00000000e+00, 8.00000000e+00, 3.00000000e+00])
""" Build and specify process configuration """

obj.update_flows(obj.variables)
obj.create_column(obj.variables)

""" variable names
     ['(HA)2(org)',	
     'H+ Extraction',
     'H+ Scrub',	
     'H+ Strip',	
     'OA Extraction',
     'OA Scrub',	
     'OA Strip', 
     'Recycle',
     'Extraction',
     'Scrub', 
     'Strip']
"""
     

""" Action methods"""

obj.update_system(obj.variables)

""" Evaluate """
obj.evaluate(obj.variables)

#print('y:', obj.y.tolist())

#print('Status:', obj.stage_status)


""" Simple reward formulation """
reward = obj.strip_pur[0] * obj.strip_recov[0]

#print(reward)
print('Purity:', obj.strip_pur[0], 'Recovery:', obj.strip_recov[0])

labels = ['(HA)2(org)', 'H+ Extraction'] # etc
indices = [0,1] # etc
values = [.3,  .001]

obj.update_var_by_label(labels, values)
obj.update_var_by_index(indices, values)

#print(obj.variables)
#print()


""" Evaluate """
obj.evaluate(obj.variables)


""" Simple reward formulation """
reward = obj.strip_pur[0] * obj.strip_recov[0]
#print(reward)

#print('recovery', obj.recovery(obj.variables))

#print(obj.y.size)

ree = ['Nd','Pr']
obj = pl.populate(obj,ree) #y0, Ny, Ns, naq,mix,header)  # Return a list of aqueous and organic dataFrames 


