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
#xtest = [0.2, 1e-05, 0.006498175348279338, 2, 3.9, 3.9, 4.35, 0.2, 5.0, 3.0, 3.0]
xtest = [0.55, 0.00833634561948692, 3.474732914376938e-05, 0.9472773186187049, 1.5, 3.15, 3.55, 0.35, 6.0, 7.0, 6]
obj.evaluate_loop(x=xtest)


""" Get variables """ #########################################################
print('Design:',obj.variables)
#= [0.5,	0.5,	0.5, 0.5,	0.5,	0.5,	0.5,	0.5,	1,	1,	1]


""" Get error status""" #######################################################
print('Status:',obj.stage_status )
#= {'Extraction-0': [True, True, True, True, True, True, True],
# 'Scrub-0': [True, True, True, True, True, True, True, True],
# 'Strip-0': [True, True, True, True]}


""" evaluate reward""" ########################################################
obj.reward()


""" Get potential objectives""" ###############################################
# Nd recovery
print('Recovery:',obj.recovery['Strip-0'][obj.ree.index('Nd')])

# Nd purity
print('Purity:',obj.purity['Strip-0'][obj.ree.index('Nd')])

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