#Most up to date version of DQAgent
# -*- coding: utf-8 -*-
'''
Created on Monday July 8, 2019
@author: Blake Richey
'''

import gym
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
from   tensorflow.keras.optimizers import Adam
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.python.client    import device_lib
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.compat.v1.set_random_seed(1)

                
class Utilities():
    """
        Utilities for agent
    """
    
    def __init__(self):
        self.aggregate_episode_rewards = {
            'min':        [],
            'max':        [],
            'epoch':      [],
            'average':    [],
            'cumulative': [],
            'loss':       [],
            'accuracy':   [],
        }

    def collect_aggregate_rewards(self, epoch, rewards, loss, accuracy):
        """Collect rewards statistics."""

        min_reward     = min(rewards)
        max_reward     = max(rewards)
        average_reward = sum(rewards)/len(rewards)
       
        self.aggregate_episode_rewards['epoch'].append(epoch)
        self.aggregate_episode_rewards['cumulative'].append(sum(rewards))

        self.aggregate_episode_rewards['min'].append(min_reward)
        self.aggregate_episode_rewards['max'].append(max_reward)        
        self.aggregate_episode_rewards['average'].append(average_reward)   

        self.aggregate_episode_rewards['loss'].append(loss) 
        self.aggregate_episode_rewards['accuracy'].append(accuracy) 
    
    def show_plots(self, version=None):
        """Show plots."""
        if version == 'cumulative':
          plt.plot(self.aggregate_episode_rewards['epoch'], \
            self.aggregate_episode_rewards['cumulative'], label="cumulative rewards")
        elif version == 'accuracy':
          plt.plot(self.aggregate_episode_rewards['epoch'], \
              self.aggregate_episode_rewards['accuracy'], label="accuracy")
        elif version == 'loss':
          plt.plot(self.aggregate_episode_rewards['epoch'], \
              self.aggregate_episode_rewards['loss'], label="loss")
        elif version == None:
            plt.plot(self.aggregate_episode_rewards['epoch'], \
              self.aggregate_episode_rewards['average'], label="average rewards")
            plt.plot(self.aggregate_episode_rewards['epoch'], \
              self.aggregate_episode_rewards['max'], label="max rewards")
            plt.plot(self.aggregate_episode_rewards['epoch'], \
              self.aggregate_episode_rewards['min'], label="min rewards")
        plt.legend(loc=4)
        plt.show()
    
    # This is a small utility for printing readable time strings:
    def format_time(self, seconds):
        if seconds < 400:
            s = float(seconds)
            return "%.1f seconds" % (s,)
        elif seconds < 4000:
            m = seconds / 60.0
            return "%.2f minutes" % (m,)
        else:
            h = seconds / 3600.0
            return "%.2f hours" % (h,)



class DQAgent(Utilities):

    def __init__(self, env, model=None, **kwargs):
        '''
            Initialize agent hyperparameters
            agent_opts = {
                #hyperparameters
                'REPLAY_BATCH_SIZE':       8,
                'LEARNING_BATCH_SIZE':     2,
                'DISCOUNT':              .99,
                'MAX_STEPS':             500,
                'REPLAY_MEMORY_SIZE':    1000,
                'LEARNING_RATE':         0.001,
                
                #ann specific
                'EPSILON_START':         .98,
                'EPSILON_DECAY':         .98,
                'MIN_EPSILON' :          0.01,
                #saving and logging results
                'AGGREGATE_STATS_EVERY':   5,
                'SHOW_EVERY':             10,
                'COLLECT_RESULTS':      False,
                'SAVE_EVERY_EPOCH':     False,
                'SAVE_EVERY_STEP':      False,
                'BEST_MODEL_FILE':      'best_model.h5',
            } 
        '''

        self.env = env
        self.weights_file  = kwargs.get('WEIGHTS_FILE',       "")

        # Hyperparameters
        self.replay_batch_size   = kwargs.get('REPLAY_BATCH_SIZE',    8)   # How many steps (samples) to use for training
        self.learning_batch_size = kwargs.get('LEARNING_BATCH_SIZE',  2)   # How many steps (samples) to apply to model.fit at a time
        self.max_steps           = kwargs.get('MAX_STEPS',          500)
        self.epsilon             = kwargs.get('EPSILON_START',      0.98)
        self.epsilon_decay       = kwargs.get('EPSILON_DECAY',      0.98)
        self.discount            = kwargs.get('DISCOUNT',           0.99)  #HIGH VALUE = SHORT TERM MEMORY
        self.replay_size         = kwargs.get('REPLAY_MEMORY_SIZE', 1000)  #steps in memory
        self.min_epsilon         = kwargs.get('MIN_EPSILON',        0.01)
        self.learning_rate       = kwargs.get('LEARNING_RATE',      0.001)

        #saving and logging results
        self.save_every_epoch  = kwargs.get('SAVE_EVERY_EPOCH',   False)
        self.save_every_step   = kwargs.get('SAVE_EVERY_STEP',    False) 
        self.best_model_file   = kwargs.get('BEST_MODEL_FILE',    'best_model.h5') #file to save best model to
        
        # Data Recording Variables
        self.collect_results       = kwargs.get('COLLECT_RESULTS',    False)
        self.show_every            = kwargs.get('SHOW_EVERY',            10)
        self.aggregate_stats_every = kwargs.get('AGGREGATE_STATS_EVERY',  5)

        # Exploration settings    

        self.explore_spec = {'EPSILON_DECAY': self.epsilon_decay,
                             'MIN_EPSILON':   self.min_epsilon}

        # Memory
        self.best_reward = {}
        self.memory      = list()

        if model:
            self.model = model
        elif self.weights_file:
            self.build_model()
            self.model = model.load_weights(self.weights_file)

        if self.collect_results:
          super().__init__()

    def build_model(self, **kwargs):
        '''
            Builds model to be trained
            model_opts = {
                'num_layers':      3,
                'default_nodes':   20,
                'dropout_rate':    0.5,
                'model_type':      'ann',
                'add_dropout':     False,
                'add_callbacks':   False,
                'nodes_per_layer': [20,20,20],
                #cnn options
                'filter_size':     3,
                'pool_size':       2,
                'stride_size':     None,
            }
        '''
        if not hasattr(self, 'model'):
            #define NN
            self.num_outputs     = self.env.action_space.n 
            self.num_layers      = kwargs.get('num_layers',      3)
            self.default_nodes   = kwargs.get('default_nodes',   20)
            self.nodes_per_layer = kwargs.get('nodes_per_layer', [])
            self.dropout_rate    = kwargs.get('dropout_rate',    0.5)
            self.add_dropout     = kwargs.get('add_dropout',     False)
            self.add_callbacks   = kwargs.get('add_callbacks',   False)
            self.model_type      = kwargs.get('model_type',      'ann')
            
            #cnn options
            self.pool_size       = kwargs.get('pool_size',        2)
            self.filter_size     = kwargs.get('filter_size',      3)
            self.stride_size     = kwargs.get('stride_size',      None)        

            self.num_features    = self.env.observation_space.shape[0]

            #Create NN
            if self.model_type == 'ann':
                assert self.num_layers >=1, \
                  'Number of layers should be greater than or equal to one!'

                self.activation    = 'linear'
                self.action_policy = 'eg'
                
                model = Sequential()
                model.add(Dense(self.num_features, input_shape = (self.num_features,)))
                
                for layer in range(self.num_layers):
        
                    try:
                        nodes=self.nodes_per_layer[layer]
                    except IndexError:
                        nodes = None

                    if nodes is None:
                        nodes = self.default_nodes

                    model.add(Dense(units = nodes, activation = 'relu'))
                    print(f'Added Dense layer with {nodes} nodes.')
                    if self.add_dropout:
                        model.add(Dropout(rate = self.dropout_rate, name='dropout_'+str(layer+1)))
                        print('Added Dropout to layer')
                
                #output layer
                model.add(Dense(units = self.num_outputs, activation = self.activation, name='dense_output'))
                model.compile(optimizer = Adam(lr=self.learning_rate), loss = 'mse', metrics=['accuracy'])
                model.summary()
            
            elif self.model_type == 'cnn':
              assert self.num_layers >=1, 'Number of layers should be greater than or equal to one!'

              self.activation     = 'softmax'
              self.action_policy  = 'softmax'
              self.envshape       = self.env.observation_space.shape
              self.batch_envshape = merge_tuple((1, self.envshape))


              model = Sequential()

              for layer in range(self.num_layers):

                try:
                  nodes=self.nodes_per_layer[layer]
                except IndexError:
                  nodes = None

                if nodes is None:
                  nodes = self.default_nodes

                if layer == 0:
                  #input layer
                  model.add(Conv2D(nodes, kernel_size=self.filter_size, activation='relu', \
                    input_shape=(self.envshape)))
                else:
                  #add hidden layers
                  model.add(Conv2D(nodes, kernel_size=self.filter_size, activation='relu'))

              model.add(MaxPooling2D(pool_size=self.pool_size, strides=self.stride_size))
              model.add(Flatten())
              #output layer
              model.add(Dense(self.num_outputs, activation='softmax'))

              #compile model using accuracy to measure model performance
              model.compile(optimizer=Adam(lr=self.learning_rate), \
                loss='categorical_crossentropy', metrics=['accuracy'])

              model.summary()             

        
        self.model = model
    
    def evaluate(self, n_epochs=1, render=True, verbose=True):
        start_time = datetime.datetime.now()
        print(f'Evaluating... Starting at: {start_time}')

        total_rewards = []
        
        for epoch in range(n_epochs):
            n_steps = 0
            done = False
            envstate = self.env.reset()
            rewards = []
            while (not done and n_steps < self.max_steps):
                prev_envstate = envstate
                if self.model_type == 'cnn':
                    q = self.model.predict(prev_envstate.reshape(self.batch_envshape))
                else:
                    q = self.model.predict(prev_envstate.reshape(1, -1))
                action        = np.argmax(q[0])
                envstate, reward, done, info = self.env.step(action)
                
                n_steps += 1
                rewards.append(reward)
                if render:
                    self.env.render()
            
            if verbose:
                dt = datetime.datetime.now() - start_time
                t = self.format_time(dt.total_seconds())            
                results = f'Epoch: {epoch}/{n_epochs-1} | ' + \
                  f'Steps {n_steps} | ' + \
                  f'Max Reward: {max(rewards)} | ' + \
                  f'Time: {t}'
                print(results)

            
            total_rewards.append(rewards)
      
        self.env.close()
        return total_rewards

    def get_batch(self):
        '''
            Gets previous states to perform a batch fitting
        '''
        mem_size   = len(self.memory)
        batch_size = min(mem_size, self.replay_batch_size)
        if self.model_type == 'cnn':
            env_size   = self.envshape
            inputs = np.zeros(( merge_tuple( (batch_size, env_size) ) ))
        else:
            env_size   = self.num_features
            inputs = np.zeros((batch_size, env_size))
        
        batch = random.sample(self.memory, batch_size)
        targets = np.zeros((batch_size, self.num_outputs))
        for i, val in enumerate(batch):
            envstate, action, reward, next_envstate, done = val

            if self.action_policy == 'softmax':
              adj_envstate = envstate.reshape(self.batch_envshape)
              adj_next_envstate = next_envstate.reshape(self.batch_envshape)
            else:
              adj_envstate = envstate.reshape(1, -1)
              adj_next_envstate = next_envstate.reshape(1, -1)

            inputs[i] = adj_envstate
            targets[i] = self.model.predict(adj_envstate)
            # targets[i] = target
            if done:
                targets[i, action] = reward
            else:
                Q_sa = np.max(self.model.predict(adj_next_envstate))
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

    def learn(self): #add callback options?
      inputs, targets = self.get_batch()
      
      callbacks = []
      if self.add_callbacks:
        callbacks = [ModelCheckpoint(filepath=self.best_model_file, monitor='loss', save_best_only=True)]

      history = self.model.fit(
          inputs,
          targets,
          callbacks = callbacks,
          batch_size = max(self.learning_batch_size, 1),
          verbose=0,
      )
      loss, accuracy = self.model.evaluate(inputs, targets, verbose=0)

      return loss, accuracy
    
    def load_weights(self, filename):
      '''loads weights from a file'''
      h5file = filename + '.h5'
      self.model.load_weights(h5file)
      print(f'Successfully loaded weights from: {h5file}')
    
    def predict(self, envstate): 
        '''
            envstate: envstate to be evaluated
            returns:  given envstate, returns best action model believes to take
              based on action policy. To be used during training, not evaluation
        '''
        assert self.model, 'Model must be present to make prediction'

        if self.action_policy == 'softmax':
            qvals = self.model.predict(envstate.reshape(self.batch_envshape))[0]
            action = np.random.choice(self.num_outputs, p=qvals)
        elif self.action_policy == 'eg': #epsilon greedy
            if np.random.rand() < self.epsilon:
                action = random.choice(range(self.env.action_space.n))
            else:
                qvals = self.model.predict(envstate.reshape(1, -1))[0]
                action = np.argmax(qvals)

        return action


    def remember(self, episode):
      'Add to replay buffer'
      envstate, action, reward, next_envstate, done = episode
      if reward > self.best_reward.get('Reward', min(reward-0.001, 0)):
        self.best_reward = {'Observation': next_envstate, 'Reward': reward}
      
      # self.memory.append(episode + [target, Q_sa])
      self.memory.append(episode)
      if len(self.memory) > self.replay_size:
        del self.memory[0]
    
    def save_weights(self, filename):
      assert self.model, 'Model must be present to save weights'
      h5file = filename + ".h5"
      self.model.save_weights(h5file, overwrite=True)
      print('Weights saved to:', h5file)
    
    def train(self, n_epochs=15000, max_steps=0, render=False):
        self.start_time    = datetime.datetime.now()
        print(f'Starting training at {self.start_time}')
        print(f'Action Decision Policy: {self.action_policy}')

        max_steps = max_steps or self.max_steps
        for epoch in range(n_epochs):
            n_steps  = 0
            done     = False
            envstate = self.env.reset()
            rewards = []
            while (not done and n_steps<max_steps):
                prev_envstate = envstate
                action = self.predict(prev_envstate)

                envstate, reward, done, info = self.env.step(action)

                episode = [prev_envstate, action, reward, envstate, done]
                self.remember(episode)

                loss, accuracy = self.learn() #fit model
                rewards.append(reward)
                n_steps += 1

                #save model if desired goal is met
                if self.save_every_step:
                    self.is_best(loss, rewards)

                if render:
                  self.env.render()

            dt = datetime.datetime.now() - self.start_time
            t  = self.format_time(dt.total_seconds())
            if epoch % self.show_every == 0:
                results = f'Epoch: {epoch}/{n_epochs-1} | ' +\
                    f'Loss: %.4f | ' % loss +\
                    f'Accuracy: %.4f | ' % accuracy +\
                    f'Steps {n_steps} | ' +\
                    f'Epsilon: %.3f | ' % self.epsilon +\
                    f'Reward: %.3f | ' % max(rewards) +\
                    f'Time: {t}'
                print(results)

            if self.collect_results and epoch % self.aggregate_stats_every == 0:
                self.collect_aggregate_rewards(epoch, rewards, loss, accuracy)
            
            #save model if desired goal is met
            if self.save_every_epoch:
                self.is_best(loss, rewards)

            #decay epsilon after each epoch
            if self.action_policy == 'eg':
                decay = self.explore_spec['EPSILON_DECAY']
                self.epsilon = max(self.min_epsilon, decay*self.epsilon)

    def is_best(self, loss, rewards):
        '''
        Used to define best results. Will most likely need to be changed
        between each environment as the goal is different for every
        environment

        Result: Saves best model to a backup file `self.best_model_file`
        '''


        if not hasattr(self, 'best_model'):
            self.best_model = {
                    'weights': self.model.get_weights(),
                    'loss':    loss,
                    'reward':   1.01
                    }

        mod_info = None
        if max(rewards) > self.best_model['reward']:
            mod_info = {
                'weights': self.model.get_weights(),
                'loss':    loss,
                'reward':   max(rewards)
            }
        elif max(rewards) == self.best_model['reward'] and loss < self.best_model['loss']:
            mod_info = {
                'weights': self.model.get_weights(),
                'loss':    loss,
            }
        
        if mod_info:
            self.best_model.update(mod_info)
            print('New best model reached: {', self.best_model['loss'], self.best_model['reward'], '}')
            self.model.save_weights(self.best_model_file, overwrite=True)
            
def merge_tuple(arr): #arr: (('aa', 'bb'), 'cc') -> ('aa', 'bb', 'cc')
  return tuple(j for i in arr for j in (i if isinstance(i, tuple) else (i,)))

#not necessary with keras. install tensorflow-gpu instead
def get_available_gpus():
  local_devices = device_lib.list_local_devices()
  return [x.name for x in local_devices if x.device_type=='GPU']