import gym
import gym_solventx 

env = gym.make('gym_solventx-v0')
envstate =  env.reset()
#observation, reward, done, _ = env.step(1)

done = False
while not done: #take action
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
