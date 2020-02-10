#Utility functions for Gym
import pandas as pd
import json

def read_config(file_name):
    """Load config json file and return dictionary."""
    
    with open(file_name, "r") as config_file:
        print(f'Reading configuration file:{config_file.name}')
        confDict = json.load(config_file)
        
    return confDict

def initialize_variable_bounds(config_file):
    """Observation bounds."""
    
    assert 'json' in config_file, 'Config file must be a json file!'
                               
    design_config = read_config(config_file)
    
    variable_config = design_config['variable_config']
    process_config = design_config['process_config']
    environment_config = design_config['environment_config']   
    
    
    logscale_min = min([variable_config['H+ Extraction']['lower'], variable_config['H+ Scrub']['lower'], variable_config['H+ Strip']['lower']])
    logscale_max = max([variable_config['H+ Extraction']['upper'], variable_config['H+ Scrub']['upper'], variable_config['H+ Strip']['upper']])
    
    #log scaled list ranging from lower to upper bounds of h+, including an out of bounds value for invalid actions consistency
    logscale     = np.array(sorted(list(np.logspace(math.log10(logscale_min), math.log10(logscale_max), base=10, num=50))\
          +[logscale_min-1]+[logscale_max+1]))
        
    return {'variable_config':variable_config,'logscale':logscale,'environment_config':environment_config}    


def create_action_dict(variable_config):
    """Create a dictionary of discrete actions."""
    """{1:{'(HA)2(org)':0.05},2:{'(HA)2(org)':-0.05}}"""
    
    action_dict = {}
    action_levels = 2
    direction = 1
    i = 1
    for key in variable_config.keys():
        for j in range(action_levels):
            action_dict[j][key] = direction*variable_config[key]['incriment'][key]
            direction = direction*-1
            i = i+1
            
    action_dict.update({0:''})
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