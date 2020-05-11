import gym, random, math, sys, os
import numpy  as np
import pandas as pd
import random 
import time

from gym.utils import seeding
from io import StringIO
from gym import error, spaces, utils, logger

import matplotlib.pyplot as plt
from solventx import solventx
from gym_solventx.envs import utilities

class SolventXEnv(gym.Env,utilities.SolventXEnvUtilities):
    """SolventX environment."""
    
    count = 0
    
    def __init__(self, config_file,identifier):
        """Creates an instance of `SolventXEnv`.
        
        Args:
           config_file (str): File name of the JSON configuration file.
           
        Raises:
          ValueError: If File name not found.
        
        """
        
        assert isinstance(config_file,str), 'Path to config file must be provided!'
        
        SolventXEnv.count = SolventXEnv.count+1 #Increment count to keep track of number of converter model instances
        self.pid = os.getpid()
        if not identifier:
           identifier = str(SolventXEnv.count)
        self.name ='SolventXEnv_'+str(self.pid) + '_' +identifier
        
        config_dict = self.get_config_dict(config_file) #Get configuration dictionary
        
        
        self.verbosity = config_dict['logging_config']['verbosity']
        self.variable_config = config_dict['variable_config']
        self.process_config = config_dict['process_config']
        self.environment_config = config_dict['environment_config']
        self.reward_config = config_dict['reward_config']
        self.logscale = config_dict['logscale']
        
        logger.set_level(eval('logger.'+self.verbosity)) #eval('logger.'+self.verbosity) #logger.DEBUG
        logger.info(f'Creating process design environment with ID:{self.name} - Logger verbosity set to {self.verbosity}!')
        
        self.setup_simulation()
        self.setup_environment()  
    
    def setup_simulation(self):
        """Setup solvenx process simulation."""
        
        logger.debug(f'{self.name}:----Setting up solvent extraction design simulation-----')
        self.sx_design = solventx.solventx(config_file=self.process_config['design_config']) # instantiate object
        self.sx_design.get_process() #Get initial values of variables and inputs
        self.strip_groups = self.get_strip_groups()
        self.sx_design.create_var_space(input_feeds=1) #Create variable space parameters
        
        logger.info(f'{self.name}:Created Solvent Extraction simulation object:{self.sx_design} for environment!')
        logger.debug(f'{self.name}: Found variable space:{self.sx_design.combined_var_space} with {len(self.sx_design.combined_var_space)} elements,Number of inputs:{self.sx_design.num_input}')
        
        """
        module_list = []
        for module,element_list in self.sx_design.target_rees.items():
            if len(element_list)==1:
                for element in self.sx_design.confDict['modules']['output']['strip']: #Extract value for each element
                    if element_list[0]==element:
                        module_list.append(module)
        print(f'Target modules:{module_list}')
        """
    
    def setup_environment(self):
        """Setup solvenx learning environment."""
        
        logger.debug('----Setting up solvent extraction learning environment-----')
        self.check_reward_config()  
        
        self.observation_dict = self.create_observation_dict(self.sx_design.combined_var_space)        
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.observation_dict),),dtype=np.float32) #Necessary for environment to work
        
        manipulated_variables =  self.get_manipulated_variables(self.sx_design.combined_var_space,self.environment_config)
        if self.environment_config['discrete_actions']:
            self.action_dict = self.create_discrete_action_dict(manipulated_variables,self.variable_config,self.environment_config)
            self.action_space = spaces.Discrete(len(self.action_dict)) #Necessary for environment to work
        else:
            self.action_dict = self.create_continuous_action_dict(manipulated_variables,self.variable_config,self.environment_config)
            lower_bound = np.array([self.action_dict[action]['min'] for action in self.action_dict])
            upper_bound = np.array([self.action_dict[action]['max'] for action in self.action_dict])
            self.action_space = spaces.Box(low=lower_bound,high=upper_bound, dtype=np.float32)   
                                    
            #Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        
        self.purity_df = pd.DataFrame() #Collects over episode
        self.recovery_df = pd.DataFrame() #Collects over episode
        self.recority_df = pd.DataFrame() #Collects over episode 
        self.design_df = pd.DataFrame() #Collects at end of episode
        
        self.episode_count = 0
        
        
    def step(self, action): #return observation, reward, done, info
        
        if not self.done: #Only perform step if episode has not ended
            self.steps += 1
            logger.debug(f'{self.name}:Taking action {action} at step:{self.steps}')
            
            prev_state = self.sx_design.x.copy()   #save previous state
            if self.environment_config['discrete_actions']:
               self.action_stats.update({action:self.action_stats[action]+1})                        
               if self.action_dict[action]: #Check if discrete action exists                  
                  self.perform_discrete_action(action) #map action to variable and action type
                  self.run_simulation()
               else:
                  logger.info(f'{self.name}:No action found in:{self.action_dict[action]}')
            else:
               self.perform_continuous_action(action) #map continuous action to variable type                 
               self.run_simulation()            
                
            if not self.convergence_failure: #Calculate reward if there is no convergence failure
                    reward = self.get_reward()                    
            else: #Replace with previous state if there is convergence failure
                    self.sx_design.x = prev_state
                    self.done   = True
                    reward = self.reward_config['min'] #-100 #Assign minimum reward if convergence failure
            
            logger.info(f'{self.name}:Completed action {action} at step {self.steps} and got reward {reward:.3f}.')
            if self.steps >= self.max_episode_steps: #Check if max episode steps reached
                    self.done = True
                    logger.warn(f'{self.name}:Maximum episode steps exceeded after {self.steps} steps - Ending episode!')                
            if all(self.design_success.values()) and not self.convergence_failure: #Check if design was successful
                    self.done = True
                    logger.warn(f'{self.name}:Design successful with recovery:{self.metric_dict["recovery"]}, purity:{self.metric_dict["purity"]} after {self.steps} steps - Ending episode!')
            if self.done:
                self.episode_end_logic()
        
        else:
            if self.convergence_failure:
                print(f'Episode completed after {self.steps} steps due to Convergence failure - Reset environment to start new simulation!')
            elif self.steps >= self.max_episode_steps:
                print(f'Episode completed after {self.steps} steps since max steps were exceeded - Reset environment to start new simulation!')
            elif all(self.design_success.values()):
                print(f'Episode completed after {self.steps} steps since design goals was met - Reset environment to start new simulation!')            
            else:
                print(f'Episode completed after {self.steps} steps due to unknown reason - Reset environment to start new simulation!')
        
        return self.sx_design.x, reward, self.done, {}
    
    def run_simulation(self):
        """Run solvent extraction simulation"""        
        
        try: #Solve design and check for convergence
            self.sx_design.evaluate_open(x=self.sx_design.x)
            
        #except (RuntimeError, ValueError, EOFError,MemoryError,ZeroDivisionError):
        except:    
            logger.error(f'{self.name}:Exception of type:{sys.exc_info()[0]} occured!')
            logger.error(f'{self.name}:Design evaluation Failed at step:{self.steps} - Terminating environment!')
            self.convergence_failure = True
        
        else: #Execute if no exception has occured
            self.check_design_convergence() #Double check convergence in all stages
    
    def check_design_convergence(self):
        """Check if solvent extraction simulation is feasible."""
        
        if not all(self.sx_design.status.values()):
            failed_modules = [stage for stage,converged in self.sx_design.status.items() if not converged]
            logger.error(f'{self.name}:Equilibrium failed at step:{self.steps} due to non-convergence in following modules:{failed_modules} - Terminating environment!')
            self.convergence_failure = True    
        else:
            converged_modules = [stage for stage,converged in self.sx_design.status.items() if converged]
            assert len(converged_modules) == len(self.sx_design.status), 'All modules should converge'
            
            logger.debug(f'{self.name}:Equilibrium succeeded at step:{self.steps} for all modules:{converged_modules}')           
    
    def perform_discrete_action(self,action):
        """Perform action"""
        
        variable_type = self.action_dict[action]['type'] #Get variable type        
        variable_index = self.action_dict[action]['index'] #Get variable index   
        variable_delta = self.action_dict[action]['delta'] #Get variable delta
        
        new_variable_value  = self.sx_design.x[variable_index] + variable_delta 
        
        #if variable_type not in self.environment_config["subdued_variables"]:
        self.update_design_variable(variable_type,variable_index,new_variable_value)
        #else:
        #    logger.debug(f'{self.name}:Not updating {variable_type} since it is in subdued variables list.')
        
    def perform_continuous_action(self,action):
        """Perform action"""
        
        for index,new_variable_value in enumerate(action):
            variable_type = self.action_dict[index]['type'] #Get variable type        
            variable_index = self.action_dict[index]['index'] #Get variable index   
            #if variable_type not in self.environment_config["subdued_variables"]:
            self.update_design_variable(variable_type,variable_index,new_variable_value)
            #else:
            #    logger.info(f'{self.name}:Not updating {variable_type} since it is in subdued variables list.')
        
    def reset(self):
        """Reset environment."""
        
        if not self.pid == os.getpid():
            logger.warn(f'PID changed from {self.pid} to {os.getpid()}!')
            self.pid == os.getpid()
            
        self.steps             = 0
        self.done              = False
       
        self.convergence_failure = False
        self.design_success =  {}
        self.invalid           = False
        
        if hasattr(self,'metric_dict'):
            del self.metric_dict
        
        self.max_episode_steps = min(self.environment_config['max_episode_steps'],self.spec.max_episode_steps)
        self.reset_simulation()
        
        if self.environment_config['discrete_actions']:
           self.action_stats = {action: 0 for action in range(self.action_space.n)}         #reset action stats
                
        #self.reward      = self.get_reward()
        #self.best_reward = self.reward

        return self.sx_design.x
   
    def reset_simulation(self):
        """Initialize design variables."""
        
        logger.debug(f'{self.name}:----Reseting simulation-----')
        if self.environment_config['random_seed']: #Check if seed is available
            logger.info(f'{self.name}:Using constant random seed {self.environment_config["random_seed"]} from config file.')
            random.seed(self.environment_config['random_seed']) #Keep same seed every episode environment should not be randomized
        else:
            random_seed = utils.seeding.create_seed()
            logger.info(f'{self.name}:Using newly generated random seed {random_seed}.')
            random.seed(random_seed)            #Randomly generate a seed in every episode
        
        logger.debug(f'{self.name}:Initial state:{self.sx_design.x}')
        for variable,index in self.observation_dict.items(): #self.sx_design.combined_var_space.items():
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
            
            if "initial_value" in self.variable_config[variable_type]:
                 random_variable_value = self.variable_config[variable_type]['initial_value']
                
            logger.debug(f'{self.name}:Initializing {variable_type} with {random_variable_value}')
            self.update_design_variable(variable_type,index,random_variable_value)
        
        self.run_simulation()
        logger.debug(f'{self.name}:Solvent extraction design evaluation converged - initialization succeeded!')        

    def update_design_variable(self,x_type,x_index,new_x_value):
        """Update design variable."""
                
        if x_type not in self.environment_config["subdued_variables"]:
            if self.variable_config[x_type]["scale"] == 'discrete':
               new_x_value = round(new_x_value)
            
            x_upper_limit = self.variable_config[x_type]['upper']
            x_lower_limit = self.variable_config[x_type]['lower']
            
            x_delta = new_x_value - self.sx_design.x[x_index]
            
            if x_delta>0.0:
                change_direction = 'Increasing'
            else:
                change_direction = 'Decreasing'
            
            logger.debug(f'{self.name}:Updating design @ step {self.steps}:{change_direction} {x_type} (index:{x_index},value:{self.sx_design.x[x_index]:0.5f},delta:{x_delta:0.5f}) to {new_x_value:0.5f} (min:{x_lower_limit},max:{x_upper_limit})')
            
            self.sx_design.x[x_index] = max(min(new_x_value,x_upper_limit),x_lower_limit) #Check limits and update variable
            
        else:
            logger.debug(f'{self.name}:Not updating design @ step {self.steps}:{x_type} is in subdued variables list.')
   
    def get_strip_groups(self):
        """Initialize design variables."""
        
        strip_groups = {}
        
        for module in self.sx_design.modules:
            strip_groups.update({module:self.sx_design.modules[module]['strip_group']})
        
        return strip_groups
   
    def get_metrics(self):
        """Extract and return metrics for each element."""       
        
        recovery = {key:value for key, value in self.sx_design.recovery.items() if key.startswith("Strip")}
        purity = {key:value for key, value in self.sx_design.purity.items() if key.startswith("Strip")}
        #strip_elements = {key:value for key, value in self.sx_design.target_rees.items() if key.startswith("Strip")}
        
        metrics = {}
        
        for metric_type in self.reward_config['metrics']: #Extract value for each metric
            metrics.update({metric_type:{}})
            if metric_type == 'recovery':
                for group in recovery:
                    metric_value = recovery[group] #Recovery  
                    metrics[metric_type].update({group:{'metric_value':metric_value}}) #,'elements':strip_elements[group]
            if metric_type == 'purity':
                for group in purity:
                    metric_value = purity[group] #Purity
                    metrics[metric_type].update({group:{'metric_value':metric_value}})                   
            if metric_type == 'recority':
                for group in recovery:
                    metric_value = recovery[group][0] * purity[group] #Recovery*Purity
                    metrics[metric_type].update({group:{'metric_value':metric_value}})    
                    logger.debug(f'{self.name}:Converted recovery {recovery[group][0]:.3f} and purity {purity[group]:.3f} from {group} into recority {metric_value:.3f}!')
                    
        return metrics #{'recovery':{'Strip-1':{'metric_value':[0.1],'elements':['Nd','Pr']}}}
    
    def get_reward(self):
        """Calculate and return reward."""
        
        rewards_per_goal = []
        self.metric_dict = self.get_metrics() #{'recovery':{'Strip-1':[0.1]}}
        
        for goal in self.environment_config['goals']:
            if goal not in self.reward_config['metrics']:
                raise ValueError(f'{goal} is not found in reward config!')
            
            metric_type= self.metric_dict[goal] #{'Strip-1':{'metric_value':[0.1],'elements':['Nd']}}
            for stage,stage_dict in metric_type.items():    
                rewards_per_stage = []
                metric_reward = 0.0
                for level,metric_config in self.reward_config['metrics'][goal]['thresholds'].items():
                    min_level = next(iter(self.reward_config['metrics'][goal]['thresholds'])) #Get the key for the first threshold
                    min_threshold = self.reward_config['metrics'][goal]['thresholds'][min_level]['threshold'] #Get the key for the first threhold
                       
                    if isinstance(stage_dict['metric_value'],list):
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
                        metric_reward = self.reward_config['metrics'][goal]['min'] #Assign minumum value if below threshold
                        
                if isinstance(metric_reward,(int,float)):
                    metric_reward = metric_reward
                elif isinstance(metric_reward,str):
                    metric_reward = eval(metric_reward)
                else:
                    raise(f'{metric_config["reward"]} is an invalid reward for stage:{stage}!')    
                
                logger.debug(f'{self.name}:Converted {goal}:{metric:.3f} from {stage} to reward:{metric_reward:.3f} using threshold {threshold_level}')
                
                rewards_per_stage.append(metric_reward) #Append reward for each stage
                self.update_design_success(goal,stage,metric)
                
            rewards_per_goal.append(np.mean(rewards_per_stage)*self.reward_config['metrics'][goal]['weight']) #Append reward for each goal -[0.4,0.9]
            
        reward = np.mean(rewards_per_goal) #Average rewards for all goals
        logger.debug(f'{self.name}:Converted rewards:{rewards_per_goal} to {reward:.3f}')
                
        if not reward >=self.reward_config['min'] and reward <=self.reward_config['max']:
            raise ValueError(f'Total reward {reward} should be between {self.reward_config["min"]} and {self.reward_config["max"]}')
            
        return reward
    
    def update_design_success(self,goal,stage,metric):
        """Check wheather design goal was achieved."""
        
        if "success_threshold" in  self.environment_config['goals'][goal]:
            if metric >= self.environment_config['goals'][goal]['success_threshold']:
                self.design_success.update({goal:True})
                logger.debug(f'{self.name}:Design was successful for {goal} in {stage} with value {metric:.3f} at step:{self.steps}.')
            else:
                self.design_success.update({goal:False})        
    
    def episode_end_logic(self):
        """Logic at end of episode."""
        
        self.episode_count = self.episode_count  +1 #Incriment episode
        recovery = {key:value[0] for key, value in self.sx_design.recovery.items() if key.startswith("Strip")}
        purity = {key:value for key, value in self.sx_design.purity.items() if key.startswith("Strip")}
        recority = {}
        
        for group in recovery:
            metric_value = recovery[group] * purity[group] #Recovery*Purity
            recority.update({group:metric_value}) 
        
        self.collect_all_metrics(recovery,purity,recority)
        self.collect_solvent_design(self.sx_design.x)
             
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
            print(f'{list(self.observation_variables.keys())[i]}:{state:.4f}')    
    
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
