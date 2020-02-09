#Utility functions for Gym
import pandas as pd




variable_types = ['(HA)2(org)',
                              'H+ Extraction',
                                         'H+ Scrub',
                                         'H+ Strip',
                                         'OA Extraction',
                                         'OA Scrub',
                                         'OA Strip', 
                                         'Recycle',
                                         'Extraction',
                                         'Scrub', 
                                         'Strip' ]
def initialize_variable_bounds(bounds_file):
    """Observation bounds."""
    
    if bounds_file:
        boundsDF = pd.read_csv(bounds_file)
        bounds = {}
        for var_name in observation_variables:
            bounds[var_name] = {'lower': boundsDF[[var_name]].iloc[0][0],
                                'upper': boundsDF[[var_name]].iloc[1][0]
                               }
    else:
        bounds = {
                '(HA)2(org)':    {'lower': 0.2,      'upper': 0.6}, 
                'H+ Extraction': {'lower': 1.00E-05, 'upper': 2  }, 
                'H+ Scrub':      {'lower': 1.00E-05, 'upper': 2  }, 
                'H+ Strip':      {'lower': 1.00E-05, 'upper': 2  }, 
                'OA Extraction': {'lower': 0.5,      'upper': 5.5}, 
                'OA Scrub':      {'lower': 0.5,      'upper': 5.5}, 
                'OA Strip':      {'lower': 0.5,      'upper': 5.5}, 
                'Recycle':       {'lower': 0,        'upper': 1  }, 
                'Extraction':    {'lower': 2,        'upper': 8  }, 
                'Scrub':         {'lower': 2,        'upper': 8  }, 
                'Strip':         {'lower': 2,        'upper': 6  },
              }

        incriment_bounds = { #increment dictionary
          '(HA)2(org)':      .05,
          'H+ Extraction':   None, #generator function, shift_value(), used instead
          'H+ Scrub':        None, #generator function, shift_value(), used instead
          'H+ Strip':        None, #generator function, shift_value(), used instead
          'OA Extraction':   .05,
          'OA Scrub':        .05,
          'OA Strip':        .05,
          'Recycle':         .05,
          'Extraction':      1,
          'Scrub':           1,
          'Strip':           1,
        }
        assert len(bounds) == incriment_bounds, 'Number of elements should be equal.'
        logscale_min = min([bounds['H+ Extraction']['lower'], bounds['H+ Scrub']['lower'], bounds['H+ Strip']['lower']])
        logscale_max = max([bounds['H+ Extraction']['upper'], bounds['H+ Scrub']['upper'], bounds['H+ Strip']['upper']])
        #log scaled list ranging from lower to upper bounds of h+, including an out of bounds value for invalid actions consistency
        logscale     = np.array(sorted(list(np.logspace(math.log10(logscale_min), math.log10(logscale_max), base=10, num=50))\
          +[logscale_min-1]+[logscale_max+1]))
        
    return {'bounds':bounds,'incriment_bounds':incriment_bounds,'logscale':logscale}    

action_dict
def create_action_dict(bounds_dict):
    """Create a dictionary of discrete actions."""
    """{1:{'(HA)2(org)':0.05},2:{'(HA)2(org)':-0.05}}"""
    
    action_dict = {}
    i = 1
    for key in bounds_dict['bounds'].keys():
        action_dict[i][key]=bounds_dict['incriment_bounds'][key]
        i = i+1
        action_dict[i][key]=-bounds_dict['incriment_bounds'][key]
    
    action_dict.update({0:0.0})
    return action_dict
        

#rounds number to the nearest value `nearest`
#`nearest` must be between 0-1
#round_nearest(1.2354, .01) -> 1.24
def round_nearest(number, nearest=.05):
  lower  = number // 1 #lower limit
  upper  = lower +1    #upper limit

  values = []          #possible values
  curVal = lower
  while curVal <= upper:
    values.append(curVal)
    curVal=truncate_number(curVal + nearest)
  
  if upper not in values:
    values.append(upper)
  
  distance = []        #distance to each possible value
  for value in values:
    distance.append(abs(number-value))
  return values[distance.index(min(distance))]


"""
def render(self, mode='human', create_graph_every=False):
    
        '''
        create_graph_every: steps per graph generation
        '''

        output = ''
        if mode == 'human':
            print(f'Action: {self.last_action}, Action Count: {self.steps}' + '\n' + \
            f'Observation: {self.envstate}' + '\n' + \
            f'Purity: {self.obj.strip_pur[0]}' + '\n' + \
            f'Recovery: {self.obj.strip_recov[0]}' + '\n' + \
            f'Reward: {self.reward}' + '\n' + \
            f'Done: {self.done}' + '\n' + \
            '========\n\n\n'
          )
        if self.done:
            output += ''.join(['=' for x in range(80)])
            output += f'\nObservation: {self.obj.variables}\n'
            output += pretty_dict(self.envstate)
            output += (f'Purity: {self.obj.strip_pur[0]}' + '\n')
            output += (f'Recovery: {self.obj.strip_recov[0]}' + '\n')
            output += f'Reward: {self.reward}'
            output += ('\n' + ''.join(['=' for x in range(80)]) + '\n\n\n')
            print(output)
            pl.populate(self.obj) 

        elif mode == 'file':
            output = f'Action: {self.last_action}, Action Count: {self.steps}' + '\n' + \
              f'Observation: {self.envstate}' + '\n' + \
              f'Purity: {self.obj.strip_pur[0]}' + '\n' + \
              f'Recovery: {self.obj.strip_recov[0]}' + '\n' + \
              f'Reward: {self.reward}' + '\n' + \
              f'Done: {self.done}' + '\n' + \
              '========\n\n\n'
        if self.done:
            output += ''.join(['=' for x in range(80)])
            output += f'\nObservation: {self.obj.variables}'
            output += pretty_dict(self.envstate)
            output += (f'Purity: {self.obj.strip_pur[0]}' + '\n')
            output += (f'Recovery: {self.obj.strip_recov[0]}' + '\n')
            output += f'Reward: {self.reward}'
            output += ('\n' + ''.join(['=' for x in range(80)]) + '\n\n\n')
            pl.populate(self.obj)

        if create_graph_every:
            if self.steps % int(create_graph_every) == 0:
                self.create_graph(render=self.done, filename=f'ss_graph{self.steps}')

        return (None, output)[mode=='file']
    
 
def create_graph(self, **kwargs):
    gen_graph(self.obj, **kwargs)
"""