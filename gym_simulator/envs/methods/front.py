# -*- coding: utf-8 -*-
"""
Front Page

Created on Mon May 20 14:11:15 2019

@author: ciloeje
"""


""" imports """
import solvent_sweep as ss


""" Initialize Column Object """
obj = ss.solvent_extraction(scale=1, feedvol = .02) # liters per minute


""" Build and specify process configuration """

obj.update_feeds(obj.variables)
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

labels = ['(HA)2(org)', 'H+ Extraction'] # etc
indices = [0,1] # etc
values = [.3,  .001]

obj.update_var_by_label(labels, values)
obj.update_var_by_index(indices, values)


""" Evaluate """
obj.evaluate(obj.variables)


""" Simple reward formulation """
reward = obj.strip_pur[0] * obj.strip_recov[0]

