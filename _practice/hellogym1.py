if __name__ == '__main__':
    import gym
    env = gym.make('Pendulum-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        obvs,fit,inf,fin = env.step(env.action_space.sample())  # take a random action
        print('observation = {}\n'\
              'reward      = {}\n'\
              'done        = {}\n'\
              'info        = {}\n'.format(obvs,fit,inf,fin))

# Theta value can get really high . gym/envs/classic_control/pendulum._get_obs: theta = 46.36041373735987

