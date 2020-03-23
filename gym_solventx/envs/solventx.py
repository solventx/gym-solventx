import gym, random, math, sys
import numpy  as np
import pandas as pd
import random 
import time

from   gym.utils import seeding
from   io        import StringIO
from   gym       import error, spaces, utils

import matplotlib.pyplot as plt
from solventx import solventx
from gym_solventx.envs import utilities,config

#from solventx.methods import solvent_sweep

class SolventXEnv(gym.Env):
    """SolventX environment."""
    
    count = 0
    
    def __init__(self, config_file=None):
        """Creates an instance of `SolventXEnv`.
        
        Args:
           config_file (str): File name of the JSON configuration file.
           
        Raises:
          ValueError: If File name not found.
        
        """
        
        assert isinstance(config_file,str), 'Path to config file must be provided!'
        
        SolventXEnv.count = SolventXEnv.count+1 #Increment count to keep track of number of converter model instances
        
        self.name              ='gym_solventx-v0'
        
        config_dict = utilities.get_config_dict(config_file) #Get bounds dict
        
        self.variable_config = config_dict['variable_config']
        self.process_config = config_dict['process_config']
        self.environment_config = config_dict['environment_config']
        self.reward_config = config_dict['reward_config']
        self.logscale = config_dict['logscale']
        
       
        
        self.goals_list        = self.environment_config['goals_list']
        self.DISCRETE_REWARD   = self.reward_config['discrete_reward']
        
        self.action_dict = utilities.create_action_dict(self.variable_config,self.environment_config)
        self.observation_variables = utilities.create_variables_list(self.variable_config,self.environment_config)       
        
        self.action_space      = spaces.Discrete(len(self.action_dict))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.observation_variables),),dtype=np.float32)        
       
    @property
    def envstate(self):
        """Observation dictionary."""
        
        return {self.observation_variables[i]: self.sx_design.x[i] for i in range(len(self.observation_variables))} 
    
    def step(self, action): #return observation, reward, done, info
        
        if not self.done: #Only perform step if episode has not ended
            self.action_stats.update({action:self.action_stats[action]+1})
            self.steps += 1
          
            if self.action_dict[action]: #Check if action exists
                prev_state = self.sx_design.variables.copy()   #save previous state
                self.perform_action(action) #map action to variable and action type
                
                try: #Solve design and check for convergence
                    self.sx_design.evaluate_loop(x=self.sx_design.variables)
                    self.sx_design.reward()
                    self.check_design_convergence()
                    
                    if not self.convergence_failure:
                        self.reward = self.get_reward()

                except:
                    print('Solvent extraction design evaluation Failed - Terminating environment!')
                    self.convergence_failure = True
                    
                if self.convergence_failure:
                    self.sx_design.x = prev_state
                    self.done   = True
                    self.reward = -100
                if self.steps >= self.max_episode_steps:
                    self.done = True
        
            else:
                print(f'No action found in:{self.action_dict[action]}')
        
        elif self.done:
            if self.convergence_failure:
                print(f'Convergence failure after {self.steps} - Reset environment to start new simulation!')
            else:
                print(f'Episode completed after {self.steps} steps - Reset environment to start new simulation!')
        
        return self.sx_design.x, self.reward, self.done, {}

    def perform_action(self,action):
        """Perform action"""
        
        variable_type = list(self.action_dict[action].keys())[0] #From Python 3.6,dict maintains insertion order by default.
        variable_delta = self.action_dict[action][variable_type]
        
        self.update_design_variable(variable_type,variable_delta)
    
    def get_design_variable(self,x_types):
        """Update design variable."""
        
        x_indexes = [list(self.observation_variables).index(x_type) for x_type in x_types]
        
        return [list(self.observation_variables)[i] for i in x_indexes]
    
    def update_design_variable(self,x_type,delta_x):
        """Update design variable."""
        
        x_upper_limit = self.variable_config[x_type]['upper']
        x_lower_limit = self.variable_config[x_type]['lower']
        x_index = list(self.observation_variables).index(x_type)  #Get variable index
        new_x_value = self.sx_design.x[x_index] + delta_x
        
        self.sx_design.x[x_index] = max(min(new_x_value,x_upper_limit),x_lower_limit) #Check limits and update variable
    
    def update_design_convergence(self):
        """Check design convergence."""
        
        if False in self.sx_design.stage_status['Extraction-0'] or False in self.sx_design.stage_status['Scrub-0'] or False in self.sx_design.stage_status['Strip-0']:
            print('Equilibrium Failed! Invalid State Reached - Terminating environment!')
            self.convergence_failure = True
    
    def reset(self):
        """Reset environment."""
        
        self.steps             = 0
        self.done              = False
       
        self.convergence_failure = False
        self.invalid           = False
        
        self.max_episode_steps = min(self.environment_config['max_episode_steps'],self.spec.max_episode_steps)
        self.action_stats = {action: 0 for action in range(self.action_space.n)}         #reset action stats
        self.sx_design = solventx.solventx(config_file=self.process_config['design_config']) # instantiate object
        x0,n_components= self.get_process() #Get initial values of variables and inputs
        self.sx_design.create_var_space(x0,n_components) #define variable space
        print(f'Var space:{self.sx_design.combined_var_space},Number of inputs:{n_components}')
        self.initialize_design_variables()
        self.sx_design.reward()
        
        self.reward      = self.get_reward()
        self.best_reward = self.reward

        return self.sx_design.x
   
    def get_process(self):
        """Get products."""
        
        #print(self.process_config,self.sx_design.confDict)
        
        input_components = self.sx_design.confDict['modules']['input']
        strip_components = self.sx_design.confDict['modules']['output']['strip']
        #extraction_components = self.sx_design.confDict['modules']['output']['extraction']
        n_components = len(input_components)
        config_key = ''
        
        print(f'Looping through following modules config:{list(config.valid_processes.keys())}')
        for key,config_dict in config.valid_processes.items():
            
           
            if set(input_components) == set(config_dict['input']):
                if set(strip_components) ==  set(config_dict['strip']):
                    
                    config_key = key
        
        if config_key:
            print(f'Found the following process config:{key}')            
        else:
            raise ValueError(f'No configuration found for input:{input_components},strip:{strip_components}!')            
       
        modules = config.valid_processes[config_key]['modules']
        x = []
       
        print(f'Process config {config_key}:Input:{input_components},Number of modules:{len(modules)}')
        print('Modules info:')
        for key,module in modules.items():
            x.extend(module['x'])
            print(f'Module {key}:{module["strip_group"]}')
        print(f'x0:{x}')
        
        return x,n_components
    
    def initialize_design_variables(self):
        """Initialize design variables."""
               
        if not self.environment_config['randomize']:
            random.seed(100) #Keep same seed every episode environment should not be randomized
        
        # Randomize initial values
        for index, variable_type in enumerate(self.observation_variables):
            variable_upper_limit = self.variable_config[variable_type]['upper']
            variable_lower_limit = self.variable_config[variable_type]['lower']
            
            if self.variable_config[variable_type]['scale']  == 'linear':
                random_variable_value = round(random.uniform(variable_lower_limit, variable_upper_limit),3)                
            elif self.variable_config[variable_type]['scale']  == 'log':
                random_variable_value = random.choice(self.logscale)                
            else:
                raise ValueError('{} is not a valid variable scale!'.format(self.variable_config[variable_type]['scale'] ))
            self.update_design_variable(variable_type,random_variable_value)
        
        #try: #Solve design and check for convergence
        self.sx_design.evaluate_loop(x=self.sx_design.x)
        print('Solvent extraction design evaluation converged - initialization succeeded!')
        #except:
            #print('Solvent extraction design evaluation failed -  initialization failed!')
        
    def render(self, mode='human', create_graph_every=False):
        '''
        create_graph_every: steps per graph generation
        '''

        output = ''
        if mode == 'human':
            print(f'Action: {self.last_action}, Action Count: {self.steps}' + '\n' + \
            f'Observation: {self.envstate}' + '\n' + \
            f'Purity: {self.sx_design.strip_pur[0]}' + '\n' + \
            f'Recovery: {self.sx_design.strip_recov[0]}' + '\n' + \
            f'Reward: {self.reward}' + '\n' + \
            f'Done: {self.done}' + '\n' + \
            '========\n\n\n'
          )
        
        return self.sx_design.x

    def show_action_stats(self,SHOW_PLOT=False):
        """Show actions count for episode."""
    
        if SHOW_PLOT:
            action_list = list(self.action_stats.keys()) 
            action_count = list(self.action_stats.values())

            x_pos = [i for i, _ in enumerate(action_list)]

            plt.bar(x_pos, action_count, color='green')
            plt.xlabel("Actions")
            plt.ylabel("Count")
            plt.title("Action totals after episode.")
            plt.xticks(x_pos, action_list)
            plt.show() 
            
        return self.action_stats

    def get_reward(self):
        reward = []
        if not self.goals_list:
          obj = self.obj
          reward = [(obj.strip_pur[obj.ree.index('Nd')] * obj.strip_recov[obj.ree.index('Nd')]) -\
            (obj.strip_pur[obj.ree.index('Pr')] * obj.strip_recov[obj.ree.index('Pr')]) +\
            (obj.scrub_pur[obj.ree.index('Pr')] * obj.scrub_recov[obj.ree.index('Pr')]) -\
            (obj.scrub_pur[obj.ree.index('Nd')] * obj.scrub_recov[obj.ree.index('Nd')]) +\
            (obj.ext_pur  [obj.ree.index('Pr')] * obj.ext_recov  [obj.ree.index('Pr')]) -\
            (obj.ext_pur  [obj.ree.index('Nd')] * obj.ext_recov  [obj.ree.index('Nd')])]
        else:
          if 'All' in self.goals_list:
            self.goals_list = ['Purity', 'Recovery', 'Stages', 'OA Extraction', 'OA Scrub', \
              'OA Strip', 'Recycle', 'Profit']

          if False in self.obj.stage_status: #invalid state
            reward.append(-25)

          else:
            # Nd recovery
            recovery = self.obj.recovery['Strip-0'][self.obj.ree.index('Nd')]
            # Nd purity
            purity = self.obj.purity['Strip-0'][self.obj.ree.index('Nd')]
            #print(recovery)
            #print(purity)
            for goal in self.goals_list:
              if goal == 'Purity':
                  if purity < self.purity_threshold:
                    if self.DISCRETE_REWARD:
                        reward.append(-1)
                    else:
                        reward.append(-abs(1-purity))
                  else:
                    if self.DISCRETE_REWARD:
                        reward.append(1)
                    else:
                        if recovery < self.recovery_threshold:
                            reward.append(0)
                        else:
                            reward.append(min(1, purity))

              if goal == 'Recovery':
                
                if self.DISCRETE_REWARD:
                  threshold     = [   0.5, 0.7, 0.8, 0.9, 0.99]
                  rec_rewards   = [0,  .5,  .7,  .8,  .9,    1] #0 if no threshold met
                  i = -1 #reward index -1
                  for index, val in enumerate(threshold):
                    if recovery > val:
                      i = index
                  reward.append(rec_rewards[1+i])
                else:
                  if recovery < self.recovery_threshold:
                    reward.append(-abs(1-recovery))
                    
                  else:
                    if purity < self.purity_threshold:
                        reward.append(0)
                    else:
                        reward.append(min(1, recovery))

              if goal == 'Stages': #lower stages = better
                col_rewards = []

                cols       = self.obj.variables.tolist()[-3:] #extraction, scrub, strip
                for i, stages in enumerate(cols):
                  if stages == 0:
                    raise ValueError('Invalid stage structure for', 
                      ['Extraction', 'Scrub', 'Strip'][i], 'column')
                  else:
                    if self.DISCRETE_REWARD:
                      bounds = self.bounds[['Extraction', 'Scrub', 'Strip'][i]]
                      lower  = bounds['lower']
                      upper  = bounds['upper']
                      norm_stages = normalize([lower, stages, upper], 0, 1)[1]

                      if norm_stages < 0.10:
                        col_rewards.append(1)
                      elif 0.10 <= norm_stages < 0.25:
                        col_rewards.append(.75)
                      elif 0.25 <= norm_stages < 0.50:
                        col_rewards.append(0.5)
                      elif 0.50 <= norm_stages < 0.75:
                        col_rewards.append(0.25)
                      else:
                        col_rewards.append(0)
                    else:
                      col_rewards.append(abs(1/stages))

                reward.append(sum(col_rewards)/3)

              if goal == 'OA Extraction': #lower = better
                bounds = self.bounds['OA Extraction']
                lower  = bounds['lower']
                upper  = bounds['upper']
                current = self.envstate['OA Extraction']

                norm_current = normalize([lower, current, upper], 0, 1)[1]
                if self.DISCRETE_REWARD:
                  if norm_current < 0.10:
                    reward.append(1)
                  elif 0.10 <= norm_current < 0.25:
                    reward.append(.75)
                  elif 0.25 <= norm_current < 0.50:
                    reward.append(0.5)
                  elif 0.50 <= norm_current < 0.75:
                    reward.append(0.25)
                  else:
                    reward.append(0)
                else:
                  reward.append(abs(1-norm_current))

              if goal == 'OA Scrub': #higher = better
                bounds = self.bounds['OA Scrub']
                lower  = bounds['lower']
                upper  = bounds['upper']
                current = self.envstate['OA Scrub']

                norm_current = normalize([lower, current, upper], 0, 1)[1]
                if self.DISCRETE_REWARD:
                  threshold       = [   0.1, 0.25, 0.5, 0.75, 0.9, .95]
                  scrub_rewards   = [0,  .1,  .25, 0.5, 0.75,  .9,   1] #0 if no threshold met
                  i = -1 #reward index -1
                  for index, val in enumerate(threshold):
                    if norm_current > val:
                      i = index
                  reward.append(scrub_rewards[1+i])
                else:
                  reward.append(norm_current)

              if goal == 'OA Strip': #higher = better
                bounds = self.bounds['OA Strip']
                lower  = bounds['lower']
                upper  = bounds['upper']
                current = self.envstate['OA Strip']

                norm_current = normalize([lower, current, upper], 0, 1)[1]
                if self.DISCRETE_REWARD:
                  threshold       = [   0.1, 0.25, 0.5, 0.75, 0.9, .95]
                  strip_rewards   = [0,  .1,  .25, 0.5, 0.75,  .9,   1] #0 if no threshold met
                  i = -1 #reward index -1
                  for index, val in enumerate(threshold):
                    if norm_current > val:
                      i = index
                  reward.append(strip_rewards[1+i])
                else:
                  reward.append(norm_current)

              if goal == 'Recycle': #lower = better
                recycle = self.envstate['Recycle']

                if self.DISCRETE_REWARD:
                  if recycle < 0.10:
                    reward.append(1)
                  elif 0.10 <= recycle < 0.25:
                    reward.append(.75)
                  elif 0.25 <= recycle < 0.50:
                    reward.append(0.5)
                  elif 0.50 <= recycle < 0.75:
                    reward.append(0.25)
                  else:
                    reward.append(0)
                else:
                  reward.append(abs(1-recycle))

              if goal == 'Profit':
                if hasattr(self.obj, 'psuedo_profit'):
                  profit = self.obj.psuedo_profit
                  max_profit = self.obj.max_profit
                  min_profit = self.obj.min_profit

                  #normalize values between 0-2, making profit second most important goal
                  norm_profit = normalize([min_profit, profit, max_profit], 0, 2)[1]

                  if self.DISCRETE_REWARD:
                    if norm_profit <= .25:
                      reward.append(-1)
                    elif .25 < norm_profit <= .5:
                      reward.append(0)
                    elif .5 < norm_profit <= .75:
                      reward.append(.25)
                    elif .75 < norm_profit <= 1:
                      reward.append(.5)
                    elif 1 < norm_profit <= 1.25:
                      reward.append(.75)
                    elif 1.25 < norm_profit <= 1.50:
                      reward.append(.85)
                    elif 1.50 < norm_profit <= 1.75:
                      reward.append(.95)
                    elif 1.75 < norm_profit:
                      reward.append(1)
                  else:
                    reward.append(norm_profit)


        #print(reward, truncate_number(sum(reward))) #logging
        return truncate_number(sum(reward)) #remove imprecision
    
    def show_design_performance(self):
        """Show the purity and recovery of current recovery."""
        
        purity = self.obj.purity['Strip-0'][self.obj.ree.index('Nd')]
        recovery = self.obj.recovery['Strip-0'][self.obj.ree.index('Nd')]
        print(f'Purity:{purity}')
        print(f'Recovery:{recovery}')
    
    def show_process_design(self,observation):
        """Shown the current design."""
        
        for i,state in enumerate(observation):
            print(f'{self.observation_variables[i]}:{state:.4f}')    
    
    def evaluate(self,agent):
        """Evaluate the environment."""
        
        done = False
        action_count = 0
        rewards = []
        observation = self.reset()
        start_time = time.time()
        while not done: #take action
            action_count += 1
            action = agent(observation)   
            self.decipher_action(action)
            
            observation, reward, done, _ = self.step(action) 
            rewards.append(reward)
            print(f'Reward:{reward}')
            self.show_design_performance()
        
        print(f'Episode time:{time.time()-start_time}')
        print(f'Final return:{sum(rewards)}')
    
    def dummy_agent(self,observation):
        """A dummy agent which takes random actions."""
        
        return random.randint(0,self.action_space.n-1)
    
    #shifts value along log scale either by inc or dec
    def shift_value(self, curVal, action):
        logscale = self.logscale
        #get new index, then shift left or right depending on action
        newIndex = (\
          max(0              , np.where(logscale == curVal)[0][0]-1),\
          min(len(logscale)-1, np.where(logscale == curVal)[0][0]+1) \
        )[action == 'inc']

        return self.logscale[newIndex]    
