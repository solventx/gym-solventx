'''
  Unit test to ensure reusability of gym_solventx
  test functions, their inputs, and their outputs

  Created 7/24/2019

  @author: Blake Richey
'''
import pytest
import numpy as np
import gym, gym_solventx

from gym_solventx.envs import solventx_env


# Check if environment can be created
def test_make():
  env = gym.make('gym_solventx-v0')
  assert env.spec.id == 'gym_solventx-v0'
  assert isinstance(env.unwrapped, solventx_env.Simulator)

#test observation space
def test_env():
  gym_spec = gym.spec('gym_solventx-v0')
  env = gym_spec.make()

  ob_space = env.observation_space
  act_space = env.action_space
  ob = env.reset()
  assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
  a = act_space.sample()
  observation, reward, done, _ = env.step(a)
  assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
  assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
  assert isinstance(done, bool), "Expected {} to be a boolean".format(done)
  
  env.close()

#test multiple actions and observations
def test_env_extended():
  for env in [gym.make('gym_solventx-v0')]:
      agent = lambda ob: env.action_space.sample()
      done = False
      ob = env.reset()
      num_steps = 0
      for _ in range(10):
        if not done:
          assert env.observation_space.contains(ob)
          action = agent(ob)
          assert env.action_space.contains(action)
          assert env.steps == num_steps
          (ob, reward, done, _) = env.step(action)
          num_steps += 1
      env.close()

#test to ensure arguements are being correctly passed to simulator
def test_arguements():
  glist = ['Recycle', 'Purity', 'Recovery']
  DISCRETE_REWARD = True
  env = gym.make('gym_solventx-v0', goals_list=glist, DISCRETE_REWARD=DISCRETE_REWARD)
  assert env.goals_list == glist
  assert env.DISCRETE_REWARD == DISCRETE_REWARD

  glist = ['OA Extraction', 'Stages', 'Profit']
  DISCRETE_REWARD = False
  env = gym.make('gym_solventx-v0', goals_list=glist, DISCRETE_REWARD=DISCRETE_REWARD)
  assert env.goals_list == glist
  assert env.DISCRETE_REWARD == DISCRETE_REWARD

#test if reset is performing required actions
def test_reset():
  gym_spec = gym.spec('gym_solventx-v0')
  env = gym_spec.make()

  ob_space = env.observation_space
  act_space = env.action_space
  ob = env.reset()
  assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
  assert env.steps == 0, 'Environment steps not reset'
  assert env.done == False, 'Environment done status not reset'
  assert env.last_action == None, 'Environment last action not reset to None'

#test if get_rewards is returning an expected type
def test_rewards():
  glist = ['Recycle', 'Purity', 'Recovery']
  DISCRETE_REWARD = True
  env = gym.make('gym_solventx-v0', goals_list=glist, DISCRETE_REWARD=DISCRETE_REWARD)
  reward = env.get_reward()
  assert isinstance(reward, float)
  assert env.reward == reward

#test get_stats() function and best_reward
def test_stats():
  env = gym.make('gym_solventx-v0')
  stats = env.get_stats()
  assert isinstance(stats, dict)
  assert env.best_reward == env.get_reward()

  stats = env.get_stats(SHOW_PLOT=True)
  assert isinstance(stats, dict)