import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class DQNAgent:
    def __init__(self, n_episodes= 1000, learning_rate= 0.001, batch_size = 32, gamma = 0.9, epsilon = 1.0, epsilon_decay = 0.999, epsilon_min = 0.1, environment='CartPole-v0'):
        #TODO: fix random seed
        self.env = gym.make(environment)
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.learning_rate = learning_rate
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0] 
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)    #stores past actions
        self.gamma = gamma                  #disount factor
        self.epsilon = epsilon              #exploration rate
        self.epsilon_decay = epsilon_decay  #shift to exploitation over time of training
        self.epsilon_min = epsilon_min      #min exploration rate

        self.output_dir = 'model_output/'+environment+'/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_model(self):
        model= Sequential()
        model.add(Dense(32, input_shape= self.env.observation_space.shape, activation='relu')) #add layer with 32 neurons
        model.add(Dense(32, activation='relu')) #add another 32 neuron layer
        model.add(Dense(self.action_size, activation='linear')) #output layer with shape of action space
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  
        if np.random.rand(1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs = 1, verbose = 0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


    def run(self):
        test_scores = []
        for e in range(self.n_episodes):
            done = False
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_size])
            total_reward = 0
            while not done:
                #if e%50 == 0:
                    #self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.observation_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
            print("episode: {}/{}, score: {}, e: {:.2}".format(e,self.n_episodes,total_reward,self.epsilon))
            if len(self.memory) > self.batch_size:
                self.replay()
            if e % 500 == 0:
                agent.save(self.output_dir + 'weights_' + '{:04d}'.format(e) + ".hdf5")
            if e % 100 == 0:
                avg_reward = self.run_test()
                test_scores.append([e,avg_reward])
                print("ran test with avg score of: {}".format(avg_reward))
        #TODO: Visualize performance on test set
        avg_reward = self.run_test()
        test_scores.append([e,avg_reward])
        print("final test with avg score of: {}".format(avg_reward))
        df = pd.DataFrame(test_scores, columns=['episode', 'avg_reward'])
        ax = sns.lineplot(x='episode', y='avg_reward', data=df)
        plt.show()
        return e 

    #run 100 test scenarios to evaulate the performance of the agent
    def run_test(self):
        #set epsilon 0 for test batch
        elsilon_temp = self.epsilon
        self.epsilon = 0.0
        total_reward = 0
        for i in range(100):
            done = False
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_size])
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.observation_size])
                state = next_state
        avg_reward = total_reward/100
        self.epsilon = elsilon_temp
        return avg_reward

if __name__ == '__main__':

    agent = DQNAgent(environment='CartPole-v0', epsilon_decay= 0.99)
    agent.run()
