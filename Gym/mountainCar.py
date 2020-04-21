import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(200):
    env.render()
    #push right every step
    env.step(2) # take a random action
env.close()