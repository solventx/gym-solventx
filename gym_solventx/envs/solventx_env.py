import gym, random, math, sys
import numpy  as np
import pandas as pd
import random 
import time

from   gym.utils import seeding
from   io        import StringIO
from   gym       import error, spaces, utils

import matplotlib.pyplot as plt
from solventx.methods import plotlyon
from solventx.methods import solvent_sweep

class SolventXEnv(gym.Env):
    """SolventX environment."""
    
    recovery_threshold = 0.15
    purity_threshold = 0.985
    
    def __init__(self, config_file=None):
        
        self.name              ='gym_solventx-v0'
        
        variable_dict = utilities.initialize_config_dict(config_file) #Get bounds dict
        self.variable_config = variable_dict['variable_config']
        self.process_config = variable_dict['process_config']
        self.environment_config = variable_dict['environment_config']
        self.logscale = variable_dict['logscale']
        
        self.goals_list        = self.environment_config['goals_list']
        self.DISCRETE_REWARD   = self.environment_config['discrete_reward']
        
        self.action_dict = utilities.create_action_dict(self.variable_config)
        self.observation_variables = self.variable_config.keys()
        
        n_actions =  len(self.action_dict)
        self.action_space      = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.observation_variables),),dtype=np.float32)        
       
        self.envstate = {variable:0.0 for variable in self.observation_variables}              

   
    def step(self, action): #return observation, reward, done, info
        
        if self.done and self.convergence_failure:
            print(f'Convergence failure after {self.steps} - Reset environment to start new simulation!')
            
        elif self.done:
            print(f'Episode completed after {self.steps} steps - Reset environment to start new simulation!')
        
        elif not self.done: #Only perform step if episode has not ended
            self.action_stats.update({action:self.action_stats[action]+1})
            self.steps += 1
          
            if self.action_dict[action]: #map action to variable and action type
                prev_state = self.sx_design.variables.copy()   #save previous state
                variable_type = list(self.action_dict[action].keys())[0] #From Python 3.6,dict maintains insertion order by default.
                variable_delta = self.action_dict[action][variable_type]
                variable_index = list(self.observation_variables).index(variable_type) 
                self.sx_design.variables[variable_index] = max(min(self.sx_design.variables[variable_index] + variable_delta,self.variable_config[variable_type]['upper']),self.variable_config[variable_type]['lower']) #update variables
            
                #determine results
                try:
                    self.sx_design.evaluate_loop(x=self.sx_design.variables)
                    self.sx_design.reward()
                    if False in self.sx_design.stage_status['Extraction-0'] or False in self.sx_design.stage_status['Scrub-0'] or False in self.sx_design.stage_status['Strip-0']:
                        print('Equilibrium Failed! Invalid State Reached - Terminating environment!')
                        self.convergence_failure = True
                        self.done   = True
                        self.reward = -100
                    else:
                        self.reward = self.get_reward()

                except:
                    print('Solvent extraction design evaluation Failed - Terminating environment!')
                    self.convergence_failure = True
                    self.done   = True
                    self.reward = -100
        
        if self.steps >= self.spec.max_episode_steps:
            self.done = True
        
        return self.sx_design.variables, self.reward, self.done, {}

    def perform_action(self,action):
        """Perform action"""
        ##To be included.
        pass

    def reset(self):
        """Reset environment."""
        
        self.steps             = 0
        self.done              = False
       
        self.convergence_failure = False
        self.invalid           = False
        self.max_episode_steps = self.spec.max_episode_steps
       
        self.sx_design = sx.solvent_extraction() # instantiate object
        
        self.sx_design.create_var_space(n_products=self.process_config['n_products'],
                                        n_components=self.process_config['n_components'],
                                        input_feeds=self.process_config['input_feeds'],) #define variable space
        variables = [0.0 for x in self.observation_variables] #Initialize all variables to zero
        
        if not self.environment_config['randomize']:
            random.seed(100) #Keep same seed every episode environment should not be randomized
            
        #reset action stats
        self.action_stats = {action: 0 for action in range(self.action_space.n)}

        # Randomize initial values
        for index, variable in enumerate(self.observation_variables):
            lower_bound  = self.variable_config[variable]['lower']
            upper_bound  = self.variable_config[variable]['upper']
                
            if self.variable_config[variable]['scale']  is 'linear':
                random_variable = random.uniform(lower, upper)
                random_variable = round(random_variable, 3)
            elif self.variable_config[variable]['scale']  is 'log':
                random_variable = random.choice(self.logscale)
                random_variable = max(min(random_variable,lower_bound),upper_bound)
              
            variables[index] = random_variable

        self.sx_design.evaluate_loop(x=variables)
        self.sx_design.reward()

        for i, var in enumerate(self.observation_variables):
            self.envstate[var] = self.sx_design.variables[i]
        self.reward      = self.get_reward()
        self.best_reward = self.reward

        return self.sx_design.variables
   
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
        
        return self.sx_design.variables

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

    def get_stats(self, SHOW_PLOT=False):
        if SHOW_PLOT:
            self.show_action_stats(SHOW_PLOT)

        return self.action_stats

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
    
    def decipher_action(self,action):
        """Map action to physical actions."""
        
        if action != 22:
            index = action//2           #variable
            action_type =  ['inc', 'dec'][action%2]  #increase/decrease
            print(f'Index:{index},Variable:{self.observation_variables[index]}-{action_type}')
        else:
            print('Do Nothing')
    
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
