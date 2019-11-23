# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:28:58 2019

@author: ciloeje
"""

import solvent_sweep as sx


""" Initialize """ ############################################################
obj = sx.solvent_extraction() # instantiate object
obj.create_var_space(n_products=2, n_components=2, input_feeds=1,) #define variable space


""" Evaluate column after action step """ #####################################
obj.evaluate_loop(x=[0.5]*8 + [1]*3)


""" Get variables """ #########################################################
obj.variables 
#= [0.5,	0.5,	0.5, 0.5,	0.5,	0.5,	0.5,	0.5,	1,	1,	1]


""" Get error status""" #######################################################
obj.stage_status 
#= {'Extraction-0': [True, True, True, True, True, True, True],
# 'Scrub-0': [True, True, True, True, True, True, True, True],
# 'Strip-0': [True, True, True, True]}


""" evaluate reward""" ########################################################
obj.reward()


""" Get potential objectives""" ###############################################
# Nd recovery
obj.recovery['Strip-0'][obj.ree.index('Nd')]

# Nd purity
obj.purity['Strip-0'][obj.ree.index('Nd')]

# Total Number of Stages
sum(obj.Ns.values())


""" Get additional recovery and purity data""" ################################
obj.recovery # dictionary of recoveries pin each column
obj.purity # dictionary of purities in each column
obj.Ns # dictionary of numbers of stages


""" Suggested bounds""" #######################################################

bounds =  {
            '(HA)2(org)':    {'lower': 0.2,      'upper': 0.6}, 
            'H+ Extraction': {'lower': 1.00E-05, 'upper': 2  }, 
            'H+ Scrub':      {'lower': 1.00E-05, 'upper': 2  }, 
            'H+ Strip':      {'lower': 1.00E-05, 'upper': 2  }, 
            'OA Extraction': {'lower': 0.5,      'upper': 5.5}, 
            'OA Scrub':      {'lower': 0.5,      'upper': 5.5}, 
            'OA Strip':      {'lower': 0.5,      'upper': 5.5}, 
            'Recycle':       {'lower': 0,        'upper': 0.95  }, 
            'Extraction':    {'lower': 2,        'upper': 12  }, 
            'Scrub':         {'lower': 2,        'upper': 12  }, 
            'Strip':         {'lower': 2,        'upper': 12  },
          }