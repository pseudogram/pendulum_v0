import gym
import numpy as np

# Use Cartpole-v0, Pendulum-v0, etc. when selecting an environment
env = gym.make('Pendulum-v0')

sin = np.sin # sin function
deg = 2*np.pi/360 # Multiply a number by deg to convert from radians to degrees

observation = env.reset()

for t in range(1000):
    env.render()
    print(t, observation)
    t_input = sin(deg*t)*1000
    action = np.ndarray((1,), buffer=np.array(t_input),dtype=float)
    print(action)
    observation, reward, done, info = env.step(action)
#    if done:
#        print("Episode finished after {} timesteps".format(t+1))
#        break


