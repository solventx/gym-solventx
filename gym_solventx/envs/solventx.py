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
from gym_solventx.envs import utilities#,config

#from solventx.methods import solvent_sweep

class SolventXEnv(gym.Env):
    """SolventX environment."""
    
    count = 0
    
    def __init__(self, config_file):
        """Creates an instance of `SolventXEnv`.
        
        Args:
           config_file (str): File name of the JSON configuration file.
           
        Raises:
          ValueError: If File name not found.
        
        """
        
        assert isinstance(config_file,str), 'Path to config file must be provided!'
        
        SolventXEnv.count = SolventXEnv.count+1 #Increment count to keep track of number of converter model instances
        
        self.name              ='gym_solventx-v0'
        
        config_dict = utilities.get_config_dict(config_file) #Get configuration dictionary
        self.logger = utilities.get_logger(config_dict,self)
        self.logger.info('Creating process design environment!')        
        
        self.variable_config = config_dict['variable_config']
        self.process_config = config_dict['process_config']
        self.environment_config = config_dict['environment_config']
        self.reward_config = config_dict['reward_config']
        self.logscale = config_dict['logscale']
        
        self.setup_simulation()
        self.setup_environment()                
       
    @property
    def envstate(self):
        """Observation dictionary."""
        
        return {self.observation_variables[i]: self.sx_design.x[i] for i in range(len(self.observation_variables))} 
    
    def setup_simulation(self):
        """Setup solvenx process simulation."""
        
        print('----Setting up solvent extraction design simulation-----')
        self.sx_design = solventx.solventx(config_file=self.process_config['design_config']) # instantiate object
        self.sx_design.get_process() #Get initial values of variables and inputs
        self.strip_groups = self.get_strip_groups()
        self.sx_design.create_var_space(input_feeds=1) #Create variable space parameters
        self.logger.debug(f'Var space:{self.sx_design.combined_var_space},Number of inputs:{self.sx_design.num_input}')
    
    def setup_environment(self):
        """Setup solvenx learning environment."""
        
        print('----Setting up solvent extraction learning environment-----')
        self.check_reward_config()        
        
        self.action_dict = utilities.create_action_dict(self.sx_design.combined_var_space,self.variable_config,self.environment_config)
        self.observation_variables = utilities.create_variables_list(self.variable_config,self.environment_config)       
        
        self.action_space      = spaces.Discrete(len(self.action_dict))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.observation_variables),),dtype=np.float32)
         
    def step(self, action): #return observation, reward, done, info
        
        if not self.done: #Only perform step if episode has not ended
            self.action_stats.update({action:self.action_stats[action]+1})
            self.steps += 1
            print(f'Starting step:{self.steps}')
            if self.action_dict[action]: #Check if action exists
                prev_state = self.sx_design.x.copy()   #save previous state
                self.perform_action(action) #map action to variable and action type
                
                self.run_simulation()
                
                if not self.convergence_failure: #Calculate reward if there is no convergence failure
                    self.reward = self.get_reward()                    
                else: #Replace with previous state if there is convergence failure
                    self.sx_design.x = prev_state
                    self.done   = True
                    self.reward = -100
                
                if self.steps >= self.max_episode_steps: #Check if max episode steps reached
                    self.done = True
        
            else:
                print(f'No action found in:{self.action_dict[action]}')
        
        else:
            if self.convergence_failure:
                print(f'Convergence failure after {self.steps} - Reset environment to start new simulation!')
            else:
                print(f'Episode completed after {self.steps} steps - Reset environment to start new simulation!')
        
        return self.sx_design.x, self.reward, self.done, {}

    
    def run_simulation(self):
        """Perform action"""
        
        """
        try: #Solve design and check for convergence
            self.sx_design.evaluate_open(x=self.sx_design.design_variables)
            self.check_design_convergence()
         
        except:
            print(f'Solvent extraction design evaluation Failed at step:{self.steps} - Terminating environment!')
            self.convergence_failure = True
        """
        self.sx_design.evaluate_open(x=self.sx_design.design_variables)      
        self.check_design_convergence()
    
    def check_design_convergence(self):
        """Perform action"""
        
        if not all(self.sx_design.status.values()):
            failed_modules = [stage for stage,converged in self.sx_design.status.items() if not converged]
            print(f'Equilibrium failed at step:{self.steps} due to non-convergence in following modules:{failed_modules} - Terminating environment!')
            self.convergence_failure = True    
        else:
            converged_modules = [stage for stage,converged in self.sx_design.status.items() if converged]
            assert len(converged_modules) == len(self.sx_design.status), 'All modules should converge'
            print(f'Equilibrium succeeded at step:{self.steps} for all modules:{converged_modules}')        
    
    def perform_action(self,action):
        """Perform action"""
        
        #variable_type = list(self.action_dict[action].keys())[0] #From Python 3.6,dict maintains insertion order by default.
        
        variable_type = self.action_dict[action]['type'] #Get variable type        
        variable_index = self.action_dict[action]['index'] #Get variable index   
        variable_delta = self.action_dict[action]['delta'] #Get variable delta
        
        new_variable_value  = self.sx_design.design_variables[variable_index] + variable_delta 
        
        self.update_design_variable(variable_type,variable_index,new_variable_value)
    
    def reset(self):
        """Reset environment."""
        
        self.steps             = 0
        self.done              = False
       
        self.convergence_failure = False
        self.invalid           = False
        
        self.max_episode_steps = min(self.environment_config['max_episode_steps'],self.spec.max_episode_steps)
        self.reset_simulation()
        
        self.action_stats = {action: 0 for action in range(self.action_space.n)}         #reset action stats
                
        self.reward      = self.get_reward()
        self.best_reward = self.reward

        return self.sx_design.x
   
    def reset_simulation(self):
        """Initialize design variables."""
        
        print('----Reseting simulation-----')
        if not self.environment_config['randomize']:
            random.seed(100) #Keep same seed every episode environment should not be randomized
        
        for variable,index in self.sx_design.combined_var_space.items():
            variable_type = variable.strip('-012') 
            variable_upper_limit = self.variable_config[variable_type]['upper']
            variable_lower_limit = self.variable_config[variable_type]['lower']
            
            if self.variable_config[variable_type]['scale']  == 'linear':
                random_variable_value = round(random.uniform(variable_lower_limit, variable_upper_limit),3)                
            elif self.variable_config[variable_type]['scale']  == 'discrete':
                random_variable_value = random.randint(variable_lower_limit, variable_upper_limit)                            
            elif self.variable_config[variable_type]['scale']  == 'log':
                random_variable_value = random.choice(self.logscale)                
            elif self.variable_config[variable_type]['scale']  == 'pH':
                random_variable_value = random.choice(self.logscale)  
            else:
                raise ValueError('{} is not a valid variable scale!'.format(self.variable_config[variable_type]['scale'] ))
            
            self.update_design_variable(variable_type,index,random_variable_value)
        
        self.run_simulation()
        #self.sx_design.evaluate_open(x=self.sx_design.design_variables)        
        self.logger.debug('Solvent extraction design evaluation converged - initialization succeeded!')        

    def update_design_variable(self,x_type,x_index,new_x_value):
        """Update design variable."""
        
        x_upper_limit = self.variable_config[x_type]['upper']
        x_lower_limit = self.variable_config[x_type]['lower']
        
        #new_x_value = self.sx_design.design_variables[x_index] + x_delta
        x_delta = new_x_value - self.sx_design.design_variables[x_index]
        
        print(f'Updating variable {x_type} (index:{x_index},current value:{self.sx_design.design_variables[x_index]:0.2f}) by {x_delta:0.2f} to get {new_x_value:0.2f}')
        
        self.sx_design.design_variables[x_index] = max(min(new_x_value,x_upper_limit),x_lower_limit) #Check limits and update variable
   
    def get_strip_groups(self):
        """Initialize design variables."""
        
        strip_groups = {}
        
        for module in self.sx_design.modules:
            strip_groups.update({module:self.sx_design.modules[module]['strip_group']})
        
        return strip_groups
   
    def get_metrics(self):
        """Extract and return metrics for each element."""
        
        """
        module_list = []
        for module,element_list in self.sx_design.target_rees.items():
            if len(element_list)==1:
                for element in self.sx_design.confDict['modules']['output']['strip']: #Extract value for each element
                    if element_list[0]==element:
                        module_list.append(module)
        print(f'Target modules:{module_list}')
        """
        recovery = {key:value for key, value in self.sx_design.recovery.items() if key.startswith("Strip")}
        purity = {key:value for key, value in self.sx_design.purity.items() if key.startswith("Strip")}
        strip_elements = {key:value for key, value in self.sx_design.target_rees.items() if key.startswith("Strip")}
        
        metrics = {}
        for metric_type in self.environment_config['goals']: #Extract value for each metric
            metrics.update({metric_type:{}})
            if metric_type == 'recovery':
                for group in recovery:
                    metric_value = recovery[group] #Recovery  
                    metrics[metric_type].update({group:{'metric_value':metric_value,'elements':strip_elements[group]}})
            if metric_type == 'purity':
                for group in purity:
                    metric_value = purity[group] #Purity
                    metrics[metric_type].update({group:{'metric_value':metric_value}})                   
        
        return metrics #{'recovery':{'Strip-1':{'metric_value':[0.1],'elements':['Nd','Pr']}}}
    
    def get_reward(self):
        """Calculate and return reward."""
        
        rewards = []
        reward_stage = 0.0
        reward_sum = 0.0
        metric_dict = self.get_metrics() #{'recovery':{'Strip-1':[0.1]}}
        
        for goal in self.environment_config['goals']:
            if goal not in self.reward_config['metrics']:
                raise ValueError(f'{goal} is not found in reward config!')
            #print('Metric dict:',metric_dict)
            metric_type= metric_dict[goal] #{'Strip-1':{'metric_value':[0.1],'elements':['Nd']}}
            reward_stage = 0.0 #Reset sum of stages for each goal
            for stage,stage_dict in metric_type.items():                
                metric_reward = 0.0
                for level,metric_config in self.reward_config['metrics'][goal].items():
                    min_level = next(iter(self.reward_config['metrics'][goal]))
                    min_threshold = self.reward_config['metrics'][goal][min_level]['threshold']
                    if 'threshold' in metric_config:
                        
                        if isinstance(stage_dict['metric_value'],list):
                            #if len(stage_dict['metric_value']) > 1:
                                #raise ValueError(f'{stage} has more than 2 elements')
                            #else:
                            metric = stage_dict['metric_value'][0]
                        elif isinstance(stage_dict['metric_value'],(float,int)):
                            metric = stage_dict['metric_value']
                        else:
                            raise ValueError(f'{type(stage_dict["metric_value"])} is an invalid metric value type!')
                        
                        if metric >= metric_config['threshold']: #Check if metric above threshold
                            threshold_level = level
                            metric_reward = metric_config['reward']
                            
                        if metric < min_threshold:
                            threshold_level = 'min'
                            metric_reward = self.reward_config['min'] #Assign maximum value if above threshold
                            
                if isinstance(metric_reward,(int,float)):
                    metric_reward = metric_reward
                elif isinstance(metric_reward,str):
                    metric_reward = eval(metric_reward)
                else:
                    raise(f'{metric_config["reward"]} is an invalid reward for stage:{stage}!')    
                #self.logger.debug(f'Converted {goal}:{metric:.3f} from {stage} to reward {metric_reward:.3f} using threshold {threshold_level}')
                print(f'Converted {goal}:{metric:.3f} from {stage} to reward {metric_reward:.3f} using threshold {threshold_level}')
                
                reward_stage = reward_stage +  metric_reward #Sum reward for each stage
                
            rewards.append(reward_stage) #Append reward for each goal -[0.4,0.9]
        #print(rewards)    
        return sum(rewards) #Sum rewards for all goals
    
    def decipher_action(self,action):
        """Perform action"""
        
        print(f'Action {action} corresponds to {self.action_dict[action]}')    
    
    def get_design_variable(self,x_types):
        """Update design variable."""
        
        x_indexes = [list(self.observation_variables).index(x_type) for x_type in x_types]
        
        return [list(self.observation_variables)[i] for i in x_indexes]
    
    def check_reward_config(self):
        """Check reward dictionary."""
        
        for goal in self.environment_config['goals']:
            min_level = next(iter(self.reward_config['metrics'][goal]))
            min_threshold = self.reward_config['metrics'][goal][min_level]['threshold']
            print(f'Minimum threshold {min_level} for {goal} is:{min_threshold}')
            
            for _,metric_config in self.reward_config['metrics'][goal].items():
                if min_threshold > metric_config['threshold']:
                    raise ValueError('Threshold for {goal}:{metric_config["threshold"]} should be greater than minimum threshold:{min_threshold}')
    
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

    def show_action_stats(self,show_plot=False):
        """Show actions count for episode."""
    
        if show_plot:
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
