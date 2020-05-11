#Utility functions for Gym
import pandas as pd
import numpy  as np
import math
import json
import logging

import gym
import gym_solventx 

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from gym import logger
from gym_solventx.envs import templates

class SolventXEnvUtilities:
    """SolventX environment."""

    def get_config_dict(self,config_file):
        """Read config file create confi dict."""
        
        assert 'json' in config_file, 'Config file must be a json file!'
        config_keys = templates.config_keys
        
        design_config = read_config(config_file)
        
        config_dict = {}
        for key in config_keys:
            if key in design_config.keys():
                config_dict.update({key:design_config[key]})
            else:
                raise ValueError(f'{key} not found in config JSON file!')
       
        variable_config= config_dict['variable_config']
        logscale_min = min([variable_config['H+ Extraction']['lower'], variable_config['H+ Scrub']['lower'], variable_config['H+ Strip']['lower']])
        logscale_max = max([variable_config['H+ Extraction']['upper'], variable_config['H+ Scrub']['upper'], variable_config['H+ Strip']['upper']])
        
        #log scaled list ranging from lower to upper bounds of h+, including an out of bounds value for invalid actions consistency
        logscale = np.array(sorted(list(np.logspace(math.log10(logscale_min), math.log10(logscale_max), base=10, num=50))\
              +[logscale_min-1]+[logscale_max+1]))
        
        config_dict.update({'logscale':logscale})   
        
        return config_dict    

    def get_manipulated_variables(self,combined_var_space,environment_config):
        """Create a dictionary of continuous actions."""
        """{0:{},1:{'type':'(HA)2(org)','index':0},2:{'type':'H+ Scrub','index':1}}"""
                
        manipulated_variables = combined_var_space.copy()
        logger.info(f'{self.name}:Following variables were found:{[j.strip("-012") for j in manipulated_variables.keys()]}')
        
        for variable in combined_var_space:
            if variable.strip('-012') not in environment_config['action_variables']: #Only keep user specified manipulated variables
                logger.info(f'Removing {variable} since it is not in action variables list.')
                del manipulated_variables[variable]
            if variable.strip('-012') in templates.constant_variables and variable.strip('-012') in manipulated_variables: #Remove constant variables
                logger.info(f'Removing {variable} since it is in constant variable list.')
                del manipulated_variables[variable]
        
        return manipulated_variables
            
    def create_continuous_action_dict(self,manipulated_variables,variable_config,environment_config):
        """Create a dictionary of continuous actions."""
        """{0:{},1:{'type':'(HA)2(org)','index':0},2:{'type':'H+ Scrub','index':1}}"""   
               
        logger.info(f'{self.name}:Creating continuous action dictionary...')
        
        continuous_action_dict = {}
        i=0        
        for variable,index in manipulated_variables.items():
            action = i
            action_variable = variable.strip('-012')  #Remove module numbers from variables list
            continuous_action_dict.update({action:{'type':action_variable,'index':index,
                                                   'min':variable_config[action_variable]['lower'],
                                                   'max':variable_config[action_variable]['upper']}})
            logger.debug(f'{self.name}:Converted {variable} into action {action}')
            i = i + 1
                        
        return continuous_action_dict
        
    def create_discrete_action_dict(self,manipulated_variables,variable_config,environment_config):
        """Create a dictionary of discrete actions."""
        """{0:{},1:{'(HA)2(org)':0.05},2:{'(HA)2(org)':-0.05}}"""
        
        n_increment_actions = environment_config['increment_actions_per_variable']
        n_decrement_actions = environment_config['decrement_actions_per_variable']
        
        total_increment_actions = n_increment_actions*len(manipulated_variables) 
        total_decrement_actions = n_decrement_actions*len(manipulated_variables)
        
        logger.info(f'Total increment actions:{total_increment_actions},Total decrement actions:{total_decrement_actions}')
        logger.info(f'{self.name}:Creating discrete action dictionary...')
       
        action_dict = {}
        action_dict.update({0:{}})    
        i = 1
        
        for variable,index in manipulated_variables.items():
            if n_increment_actions>0:
                for j in range(1,n_increment_actions+1):
                    action_variable = variable.strip('-012')  #Remove module numbers from variables list
                    if variable_config[action_variable]['scale'] == 'linear':
                        delta_value = j*variable_config[action_variable]['delta']
                    elif variable_config[action_variable]['scale'] == 'discrete':
                        delta_value = int(j*variable_config[action_variable]['delta'])                
                    elif variable_config[action_variable]['scale'] == 'log':
                        delta_value = 10**(j*variable_config[action_variable]['delta']) #Convert log to actual number
                    elif variable_config[action_variable]['scale'] == 'pH':
                        delta_value = 10**(-j*variable_config[action_variable]['delta']) #Convert pH to actual number
                    
                    else:
                        raise ValueError(f'{variable_config[action_variable]["scale"]} is an invalid scale for {action_variable} in increment action!')
                    
                    action_dict.update({i:{'type':action_variable,'delta':delta_value,'index':index}})
                    logger.debug(f'{self.name}:Converted incriment {action_dict[i]["delta"]:.2f} ({variable_config[action_variable]["scale"]} scale) for variable {action_variable} into action {i}')
                    i = i+1
                    
            if n_decrement_actions>0:
                for k in range(1,n_decrement_actions+1):
                    if variable_config[action_variable]['scale'] == 'linear':
                        delta_value = -k*variable_config[action_variable]['delta']
                    elif variable_config[action_variable]['scale'] == 'discrete':
                        delta_value = int(-k*variable_config[action_variable]['delta'])        
                    elif variable_config[action_variable]['scale'] == 'log':
                        delta_value = -10**(k*variable_config[action_variable]['delta']) #Convert log to actual number
                    elif variable_config[action_variable]['scale'] == 'pH':
                        delta_value = -10**(-k*variable_config[action_variable]['delta']) #Convert pH to actual number                
                    else:
                        raise ValueError(f'{variable_config[action_variable]["scale"]} is an invalid scale for {action_variable} in decrement action!')                
                    action_dict.update({i:{'type':action_variable,'delta':delta_value,'index':index}})
                    logger.debug(f'{self.name}:Converted decriment {action_dict[i]["delta"]:.2f} ({variable_config[action_variable]["scale"]} scale) for variable {action_variable} into action {i}')                
                    i = i+1
        
        return action_dict

    def create_observation_dict(self,combined_var_space):
        """Create a list of all design variables in every stage."""
        
        observed_var_space = combined_var_space.copy()
        for variable in templates.constant_variables: #Remove constant variables
            if variable in observed_var_space:
                del observed_var_space[variable]
        
        logger.info(f'Following observation variables were found:{list(observed_var_space.keys())}')
        
        return observed_var_space
    
    def check_reward_config(self):
        """Check reward dictionary."""
        
        reward_weights = []
        
        for goal in self.environment_config['goals']:
            min_level = next(iter(self.reward_config['metrics'][goal]['thresholds']))
            min_threshold = self.reward_config['metrics'][goal]['thresholds'][min_level]['threshold']
            logger.debug(f'Minimum threshold {min_level} for {goal} is:{min_threshold}')
            
            for _,metric_config in self.reward_config['metrics'][goal]['thresholds'].items():
                if min_threshold > metric_config['threshold']:
                    raise ValueError('Threshold for {goal}:{metric_config["threshold"]} should be greater than minimum threshold:{min_threshold}')
            
            reward_weights.append(self.reward_config['metrics'][goal]['weight'])
        
        if not math.isclose(np.mean(reward_weights), 1.0,abs_tol=0.001):
            raise ValueError(f'Mean of the reward weights is {np.mean(reward_weights):.3f} which is greater that 1.0!')            
    
    def collect_all_metrics(self,recovery,purity,recority):
        """Check reward dictionary."""
        
        self.collect_purity_metric(purity)
        self.collect_recovery_metric(recovery)    
        self.collect_recority_metric(recority)
    
    def collect_purity_metric(self,purity):
        """Collect purity in a dataframe."""
        
        logger.debug(f'{self.name}:Collecting purity at end of episode {self.episode_count}')
        self.purity_df = self.purity_df.append(purity, ignore_index=True)        
   
    def collect_recovery_metric(self,recovery):
        """Collect recovery in a dataframe."""
        
        logger.debug(f'{self.name}:Collecting recovery at end of episode {self.episode_count}')
        self.recovery_df = self.recovery_df.append(recovery, ignore_index=True)        
   
    def collect_recority_metric(self,recority):
        """Collect recority in a dataframe."""
        
        logger.debug(f'{self.name}:Collecting recority at end of episode {self.episode_count}')
        self.recority_df = self.recority_df.append(recority, ignore_index=True)   
    
    def collect_solvent_design(self,x):
        """Collect recority in a dataframe."""
        
        logger.debug(f'{self.name}:Collecting design at end of episode {self.episode_count}')
        
        design_dict = {key:self.sx_design.x[index] for key, index in self.observation_dict.items()}
        
        self.design_df = self.design_df.append(design_dict, ignore_index=True)  
    
    def show_metric_statistics(self):
        """Show metric statistics."""
        
        print(f'Recovery statistics after {self.episode_count} episodes:')
        print(self.recovery_df.describe())
        print(f'Purity statistics after {self.episode_count} episodes:')
        print(self.purity_df.describe())
        print(f'Recority statistics after {self.episode_count} episodes:')
        print(self.recority_df.describe())
    
    def show_solvent_design(self):
        """Show metric statistics."""
        
        print(f'Solvent design over {self.episode_count} episodes:')
        print(self.design_df)
        print(f'Solvent design statistics after {self.episode_count} episodes:')
        print(self.design_df.describe())
    
    
    def decipher_action(self,action):
        """Perform action"""
        
        if isinstance(action,(int,float)):
           print(f'{self.name}:Action {action} corresponds to {self.action_dict[action]}')        
        else:
           print(f'{self.name}:Action {action} corresponds to {self.action_dict}')
    
    def print2terminal(self,text_string):
        """Print information based on verbosity."""
        
        if self.verbosity == 'DEBUG':
           print(text_string)        
        elif self.verbosity == 'INFO':
           pass
        else:
            raise ValueError(f'{self.verbosity} is an invalid verbosity keyword!')
   

def read_config(file_name):
    """Load config json file and return dictionary."""
    
    with open(file_name, "r") as config_file:
        print(f'Reading configuration file:{config_file.name}')
        confDict = json.load(config_file)
        
    return confDict

def get_logger(config_dict,instance):
    """Initialize logger."""
    
    logger = logging.getLogger (__name__)#(type(instance).__name__)
    logging_level = config_dict['logging_config']['verbosity']
    logger.setLevel(eval('logging.'+logging_level)) 
    print(f'Logging level:{logger.level}')    
    return logger

def get_tf_env(env_name,config_file):
  """Get TF environment."""
  
  print(f'Loading environment:{env_name}')
  gym_env = gym.make(env_name, config_file=config_file)
  py_env = suite_gym.wrap_env(gym_env)
  tf_env = tf_py_environment.TFPyEnvironment(py_env) 
  
  return tf_env  
    
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



#removes imprecision issues with floats: 1-.1 = .900000000001 => 1-.1 = .9
def truncate_number(f_number, n_decimals=6):
  strFormNum = "{0:." + str(n_decimals+5) + "f}"
  trunc_num  = float(strFormNum.format(f_number)[:-5])
  return(trunc_num)

def pretty_dict(D):
  string = ''
  for key in D:
    string += (str(key) + ': ' + str(D[key]) + '\n')
  return string



#normalizes a set of data between a range
def normalize(data, rangeMin, rangeMax):
  if(rangeMin>rangeMax):
    raise ValueError('Invalid Ranges')
  newVals = []
  maxVal=max(data)
  minVal=min(data)
  for val in data:
    if maxVal-minVal == 0:
      newVals.append(rangeMin)
    else: 
      newVals.append((rangeMax-rangeMin)*(val-minVal)/(maxVal-minVal)+rangeMin)
  return newVals

def silence_function(func, *args, **kwargs):
  '''
    Replaces stdout temporarily to silence print statements inside a function
  '''
  #mask standard output
  actualstdout = sys.stdout
  sys.stdout   = StringIO()

  try:
    func(*args, **kwargs)
  finally: #set stdout but dont catch error
    sys.stdout = actualstdout



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


 #determines if the new state value would be valid before updating state
    def isValid(self, var_name, newVal):
        bounds = self.bounds[var_name]
        lower  = bounds['lower']
        upper  = bounds['upper']

        if lower <= newVal <= upper:
            return True

        return False

      #updates relevant variables after an action is performed
    def update_env(self, modInfo):
        if modInfo:
            self.envstate.update(modInfo)
"""
