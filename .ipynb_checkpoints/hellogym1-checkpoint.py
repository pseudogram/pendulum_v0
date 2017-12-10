import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(1000):
    env.render()
    obvs,fit,inf,fin = env.step(env.action_space.sample())  # take a random action
    print('observation = {}\n'\
          'reward      = {}\n'\
          'done        = {}\n'\
          'info        = {}\n'.format(obvs,fit,inf,fin))
