# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:49:24 2020

@author: splathottam
"""

import rbfopt
import pandas as pd
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from solventx import solventx as sx
from solventx import utilities as util
from gym_solventx.envs import env_utilities,templates

def get_policy(policy_location):
    """Load policy"""
  
    print(f'Loading policy at {policy_location}')
    #return tf.compat.v2.saved_model.load(policy_location)
    return tf.saved_model.load(policy_location)


def compare_agent_with_optimization(process_config_file,variable_config_file,env_config_file,policy_dir,policy_file,n_cases = 10):
    """Compare agent with optimization"""    

    env_name = 'gym_solventx-v0'
    
    print(f'Generating {n_cases} cases for feed input...')
    cases = util.generate(util.read_config(process_config_file),n_cases)
    
   
    agent_results_df = agent_evaulation_loop_with_cases(env_name,env_config_file,os.path.join(policy_dir,policy_file),cases)
    optimization_results_df = optimization_evaluation_loop(process_config_file,variable_config_file,cases)
    
    print(agent_results_df)
    print(optimization_results_df)    
    
    comparison_df = pd.concat([agent_results_df,optimization_results_df], axis=1, sort=False)
    comparison_df.to_csv('comparison_df.csv')
    
    for strip_stage in templates.strip_stages:
        if templates.agent_purity+strip_stage in comparison_df.columns:
            print(f'{templates.agent_purity+strip_stage} was found in dataframe....')
            print('Filtering....')
            filtered_comparison_df = comparison_df.loc[comparison_df[templates.agent_purity+strip_stage] >= 0.985]
            filtered_comparison_df = filtered_comparison_df.loc[filtered_comparison_df[templates.optim_purity+strip_stage] >= 0.985]
            print('Plotting box plots....')
            make_boxplot(comparison_df,[templates.agent_purity+strip_stage,templates.optim_purity+strip_stage])
            make_boxplot(comparison_df,[templates.agent_recovery+strip_stage,templates.optim_recovery+strip_stage])
            print('Plotting filtered box plots....')
            make_boxplot(filtered_comparison_df,[templates.agent_purity+strip_stage,templates.optim_purity+strip_stage])
            make_boxplot(filtered_comparison_df,[templates.agent_recovery+strip_stage,templates.optim_recovery+strip_stage])
               
            print('Plotting distribution...')   
            make_distributionplot(comparison_df,[templates.agent_purity+strip_stage,templates.optim_purity+strip_stage])
            make_distributionplot(comparison_df,[templates.agent_recovery+strip_stage,templates.optim_recovery+strip_stage])
           
    #filtered_comparison_df.drop(columns=['Nd.1','Pr.1'], inplace=True)
    filtered_comparison_df.to_csv('filtered_comparison_df.csv')   
    
   
def agent_evaulation_loop_with_cases(env_name,env_config_file,policy_file,cases):
    """Test tf-agent policy"""    
  
    tf_env = env_utilities.get_tf_env(env_name,env_config_file)
    policy = get_policy(policy_file)

    print(f'Policy type:{type(policy)}!')  
    print(f'Testing for {len(cases)} feed input cases!')
  
    returns = []
    recovery_list = []
    purity_list = []
    
    t1 = time.time()
    for case_key,case in cases.items():
        episode_return = evaluate_with_agent(tf_env,policy,case)
        returns.append(episode_return.numpy())
        
        recovery = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.recovery.items() if key.startswith("Strip")}
        purity = {key:value for key, value in tf_env._env._envs[0]._gym_env.env.sx_design.purity.items() if key.startswith("Strip")}

        print(f'Feed conc @ case {case_key}:{case}')
        print(f'Recovery at episode {case_key}:{recovery}')
        print(f'Purity at episode {case_key}:{purity}')
        print(f'Design success at episode {case_key}:{tf_env._env._envs[0]._gym_env.env.design_success}')
        print(f'Total return at episode {case_key}:{episode_return}')
            
        recovery_list.append(recovery)
        purity_list.append(purity)
   
    t2 = time.time()

    print(f'Total Time:{t2-t1:.2f}')
    print(f'Average time per case:{(t2-t1)/len(cases):.2f}')
    
    print(f'List of returns after {len(cases)} episodes:{returns}')
    print(f'Average return:{np.mean(returns):.3f},Standard deviation:{np.std(returns):.3f}')
    tf_env._env._envs[0]._gym_env.env.show_all_initial_metrics()
    tf_env._env._envs[0]._gym_env.env.show_all_final_metrics()
    tf_env._env._envs[0]._gym_env.env.show_initial_metric_statistics()
    tf_env._env._envs[0]._gym_env.env.show_final_metric_statistics()
    tf_env._env._envs[0]._gym_env.env.show_initial_design()
    tf_env._env._envs[0]._gym_env.env.show_final_design()
    print(f'Solvent extraction state:{tf_env._env._envs[0]._gym_env.env.sx_design.x}')
    tf_env._env._envs[0]._gym_env.env.save_metrics()
    tf_env._env._envs[0]._gym_env.env.save_design()
    elements = tf_env._env._envs[0]._gym_env.env.elements
    
    results_df = pd.concat([tf_env._env._envs[0]._gym_env.env.final_design_df,tf_env._env._envs[0]._gym_env.env.final_recovery_df,tf_env._env._envs[0]._gym_env.env.final_purity_df], axis=1, sort=False) 

    return results_df   

def evaluate_with_agent(tf_env,policy,case):
    """Test agent"""
    
    tf_env._env._envs[0]._gym_env.env.update_state_dict(case)
    time_step = tf_env.reset()
    policy_state = policy.get_initial_state(tf_env.batch_size)
    print(f'Initial Time step:\n{time_step}')
    print(f'Initial policy state:{policy_state}!')
    
    step = 0    
    episode_return = 0.0
    
    while not time_step.is_last():
        action_step = policy.action(time_step,policy_state)
        time_step = tf_env.step(action_step.action)
        print(f'Step:{step}:Reward:{time_step.reward},Observation:{time_step.observation}')
              
        episode_return += time_step.reward
        step = step + 1
    
    return episode_return

def optimization_evaluation_loop(config_file,config_env_file,cases):
    """Test agent"""
    
    confDict = util.read_config(config_file)    
    confEnvDict = util.get_env_config_dict(config_env_file)
    
    logging_interval = 5
    iters = 100
    
    results_df = pd.DataFrame()
    design_df = pd.DataFrame()
    
    t1 = time.time()

    for case_key,case in cases.items():
        #resjsn = evaluate_with_optimization(confDict, confEnvDict, case)
        
        ree_mass = [item for item in case.values()]
        sx_design = sx.solventx(confDict, confEnvDict, ree_mass) # instantiate solvent extraction object
        sx_design.cases = case

        resjsn = util.optimize(sx_design, iters)
        #print(resjsn)
    
        print(f'\nFeed conc @ case {case_key}:{case}')
        print(f'Recovery @ case {case_key}:{resjsn["recovery"]}')
        print(f'Purity @ case {case_key}:{resjsn["purity"]}')
        print(f'Recority @ case {case_key}:{resjsn["objective"]}')
        
        recovery_dict = {}
        purity_dict = {}
        
        for key,item in resjsn["recovery"].items():
            if 'Strip' in key:
                recovery_dict.update({templates.optim_recovery+key:item[0]}) 
        for key,item in resjsn["purity"].items():
            if 'Strip' in key:
                purity_dict.update({templates.optim_purity+key:item}) 

        results_df = results_df.append({**case,**recovery_dict,**purity_dict}, ignore_index=True,sort=True) 
        
        design_df = design_df.append({**resjsn["design"]},ignore_index=True,sort=True)
    
        if eval(case_key)%logging_interval == 0:
            print(f'Saving dataframe at case:{case_key}...')
            results_df.to_csv('optim_results.csv')
    t2 = time.time()

    results_df.to_csv('optim_results.csv')
    design_df.to_csv('optim_results_design.csv')
    
    print(f'Total Time:{t2-t1:.2f}')
    print(f'Average time per case:{(t2-t1)/len(cases):.2f}')
    
    return results_df
        
def make_boxplot(results_df,quantities):
    """Test agent""" 
    
    boxplot = results_df.boxplot(column=quantities)
    plt.show()
    
def make_distributionplot(results_df,quantities):
    """Test agent""" 
    
    targets = [results_df[quantity] for quantity in quantities]
    
    for target in targets:
        
        if 'recovery' in target.name:
            quantity = 'recovery'
        if 'purity' in target.name:
            quantity = 'purity'
        if 'agent' in target.name:
            agent_type = 'agent'
        if 'optim' in target.name:
            agent_type = 'optim'
           
        sns.distplot(target,hist=True, rug=True,axlabel=quantity,label=agent_type)
    plt.legend()
    plt.show()    
