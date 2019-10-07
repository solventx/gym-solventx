import gym, random, math, sys
import numpy  as np
import pandas as pd

from   gym.utils import seeding
from   io        import StringIO
from   gym       import error, spaces, utils

import matplotlib.pyplot                        as plt
import gym_solventx.envs.methods.plotlyon      as pl
import gym_solventx.envs.methods.solvent_sweep as ss
from   gym_solventx.envs.methods.ssgraph       import create_graph as gen_graph

class SolventXEnv(gym.Env):

  def __init__(self, DISCRETE_REWARD=None, goals_list=None, bounds_file=None):
    self.steps             = 0
    self.last_action       = None
    self.done              = False
    self.goals_list        = goals_list
    self.DISCRETE_REWARD   = DISCRETE_REWARD
    self.name              ='gym_solventx-v0'
    self.obj               = ss.solvent_extraction(scale=1, feedvol = .02)

    self.action_space      = spaces.Discrete(23)
    self.action_stats      = {action_num: 0 for action_num in range(self.action_space.n)}
    self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.obj.variables),),dtype=np.float32)
    self.observation_variables = [
     '(HA)2(org)',	
     'H+ Extraction',
     'H+ Scrub',	
     'H+ Strip',	
     'OA Extraction',
     'OA Scrub',	
     'OA Strip', 
     'Recycle',
     'Extraction',
     'Scrub', 
     'Strip'
    ]

    if bounds_file:
      boundsDF = pd.read_csv(bounds_file)
      self.bounds = {}
      for var_name in self.observation_variables:
        self.bounds[var_name] = {
          'lower': boundsDF[[var_name]].iloc[0][0],
          'upper': boundsDF[[var_name]].iloc[1][0]
        }
    else:
      self.bounds = {
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

    self.inc_dict = { #increment dictionary
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

    logscale_min = min([self.bounds['H+ Extraction']['lower'], self.bounds['H+ Scrub']['lower'], self.bounds['H+ Strip']['lower']])
    logscale_max = max([self.bounds['H+ Extraction']['upper'], self.bounds['H+ Scrub']['upper'], self.bounds['H+ Strip']['upper']])
    #log scaled list ranging from lower to upper bounds of h+, including an out of bounds value for invalid actions consistency
    self.logscale     = np.array(sorted(list(np.logspace(math.log10(logscale_min), math.log10(logscale_max), base=10, num=50))\
      +[logscale_min-1]+[logscale_max+1]))

    self.envstate = dict()
    for i, var in enumerate(self.observation_variables):
      self.envstate[var] = self.obj.variables[i]

    variables = self.obj.variables
    self.obj.update_flows(variables)
    self.obj.create_column(variables)

    silence_function(self.obj.evaluate, self.obj.variables)
    self.invalid     = False
    self.reward      = self.get_reward()
    self.best_reward = self.reward

  def step(self, action): #return observation, reward, done, info
    if not self.done:
      self.last_action = action
      self.action_stats[action]+=1
    
    if action == self.action_space.n-1 or self.done: #last action = do nothing
      self.invalid = False
      self.reward  = self.get_reward()
      self.update_env(None) #update actions count, but not state
    else:
      #map action to variable and action type
      index, action_type = action//2, ['inc', 'dec'][action%2] # variable, increase/decrease
      variables        	 = self.obj.variables
      var_name         	 = self.observation_variables[index]

      #perform actions
      mod_info           = self.perform_action(index, action_type)

      #update variables
      prev_state = variables.copy()
      variables[index] = mod_info[var_name]
      self.update_env(mod_info)
      self.obj.update_system(variables)

      #determine results
      if not np.array_equal(prev_state, variables): #if state has changed
        try:
          silence_function(self.obj.evaluate, self.obj.variables)
          if False in self.obj.stage_status:
            print('Equilibrium Failed! Invalid State Reached!')
          self.reward = self.get_reward()
        except:
          print('Epoch Failed!')
          self.done   = True
          self.reward = -100
    
    return self.obj.variables, self.reward, self.done, None

  def perform_action(self, index, action):
    variables = self.obj.variables
    var_name  = self.observation_variables[index]
    inc       = self.inc_dict[var_name]

    if action == 'inc':
      if type(inc) == int:
        newVal = int(variables[index] + inc)
      elif type(inc) == float:
        newVal = truncate_number(variables[index] + inc)
      else:
        newVal = self.shift_value(variables[index], 'inc')
    elif action == 'dec':
      if type(inc) == int:
        newVal = int(variables[index] - inc)
      elif type(inc) == float:
        newVal = truncate_number(variables[index] - inc)
      else:
        newVal = self.shift_value(variables[index], 'dec')

    if not self.isValid(var_name, newVal): #invalid state value attempted
      self.invalid = True
      bounds = self.bounds[var_name]
      lower  = bounds['lower']
      upper  = bounds['upper']

      if newVal > upper:
        newVal = upper
      elif newVal < lower:
        newVal = lower
    else:
      self.invalid = False
    
    return {var_name: newVal}
    
  def reset(self):
    self.steps             = 0
    self.last_action       = None
    self.done              = False
    self.invalid           = False
    self.max_episode_steps = self.spec.max_episode_steps
    self.obj               = ss.solvent_extraction(scale=1, feedvol = .02)

    #reset action stats
    for key in self.action_stats:
      self.action_stats[key] = 0

    # Randomize initial values
    for index, var in enumerate(self.observation_variables):
      inc    = self.inc_dict[var]
      bounds = self.bounds[var]
      lower  = bounds['lower']
      upper  = bounds['upper']
      if inc:
        rand = random.uniform(lower, upper)
        rand = round_nearest(rand, inc)
      else:
        rand = random.choice(self.logscale)
      if rand < lower:
        rand = lower
      elif rand > upper:
        rand = upper
      self.obj.variables[index] = rand

    self.envstate = dict()
    for i, var in enumerate(self.observation_variables):
      self.envstate[var] = self.obj.variables[i]

    self.feed[0] = random.uniform(self.fbounds[0],self.fbounds[1]) #Randomize Nd concentration within bounds
    self.feed[1] = self.feed_total - self.feed[0]  #Randomize Pr concentartion
        
    variables = self.obj.variables
    self.obj.update_flows(variables)
    self.obj.create_column(variables)

    try:
      silence_function(self.obj.evaluate, self.obj.variables)  
    except:
      return self.reset()
    self.reward      = self.get_reward()
    self.best_reward = self.reward

    
    return self.obj.variables


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
    if not self.done:
      self.steps += 1
      if self.steps >= self.max_episode_steps:
        self.done = True
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
        for goal in self.goals_list:
          if goal == 'Purity':
              purity = self.obj.strip_pur[0]
              #print(purity)
              if purity < .985:
                if self.DISCRETE_REWARD:
                  reward.append(-1)
                else: 
                  reward.append(-abs(1-purity))
              else:
                if self.DISCRETE_REWARD:
                  reward.append(1)
                else:
                  reward.append(min(1, purity))

          if goal == 'Recovery':
            recovery  = self.obj.strip_recov[0]
            #print(recovery)

            if self.DISCRETE_REWARD:
              threshold     = [   0.5, 0.7, 0.8, 0.9, 0.99]
              rec_rewards   = [0,  .5,  .7,  .8,  .9,    1] #0 if no threshold met
              i = -1 #reward index -1
              for index, val in enumerate(threshold):
                if recovery > val:
                  i = index
              reward.append(rec_rewards[1+i])
            else:
              if recovery < 0.50:
                reward.append(-pow((1+abs(1-recovery)), 4))
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
