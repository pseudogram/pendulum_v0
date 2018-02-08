import gym

def real_world():
    ENV_NAME = 'Pendulum-v0'
    EPISODES = 3  # Number of times to run envionrment when evaluating
    STEPS = 200  # Max number of steps to run run simulation

    env = gym.make(ENV_NAME)

    # Used to create controller
    obs_dim = env.observation_space.shape[0]  # Input to controller (observ.)
    action_dim = env.action_space.shape[0]  # Output from controller (action)
    nodes = 10  # Unconnected nodes in network in the
    # dt = 0.05  # dt for the environment, found in environment source code
    dt = 1 # Works the best, 0.05 causes it to vanish
    return env, obs_dim,action_dim


# class NARKSSimulator:
