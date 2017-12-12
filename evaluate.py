import numpy as np
from new_rnn import Controller
import gym
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

ENV_NAME = 'Pendulum-v0'
EPISODES = 10  # 100000
STEPS = 200
TEST = 100

env = gym.make(ENV_NAME)


def evaluate():

    # Create controller
    state_dim = env.observation_space.shape[0]  # Input to controller (observ.)
    action_dim = env.action_space.shape[0]  # Output from controller (action)
    nodes = 6  # Unconnected nodes in network in the
    # TODO: ASK CHRIS ABOUT dt, may need to optimize without knowing dt
    dt = 0.05  # Time between steps in env.  may be unrelated to dt in RNN?



    agent = Controller(state_dim,action_dim,nodes=nodes,dt=dt)

    # agent._init_vars()

    total_reward = 0
    for episode in range(EPISODES):
        print("Starting Episode: {}".format(episode))

        # Starting observation
        observation = env.reset()

        episode_reward = 0
        next_state = np.ones([agent.state_size, 1], dtype=np.float32)
        for step in range(STEPS):
            # env.render()
            observation = np.reshape(observation,(3,1))
            # print(observation.shape)
            action, next_state = agent.percieve(observation, next_state)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # print('step',step, 'action', action, 'observation', observation)

            if done:
                break

        print("Episode {} = {}".format(episode, episode_reward))
        total_reward += episode_reward

    print(agent.get_weights().eval())

    # returns the average reward for number of episodes run
    total_reward /= EPISODES
    return total_reward


def test(agent):

    total_reward = 0
    for episode in range(EPISODES):
        print("Starting Episode: {}".format(episode))

        # Starting observation
        observation = env.reset()

        episode_reward = 0
        next_state = np.ones([agent.state_size, 1], dtype=np.float32)
        for step in range(STEPS):
            env.render()
            observation = np.reshape(observation,(3,1))
            # print(observation.shape)
            action, next_state = agent.percieve(observation, next_state)
            print(action)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # print('step',step, 'action', action, 'observation', observation)

            if done:
                break

        # print("Episode {} = {}".format(episode, episode_reward))
        total_reward += episode_reward

    # print(agent.get_weights().eval())

    # returns the average reward for number of episodes run
    total_reward /= EPISODES
    return total_reward


def upload():
    # # upload result
    # env.monitor.start('gym_results/Pendulum-v0-experiment-1', force=True)
    # for i in xrange(100):
    #     total_reward = 0
    # state = env.reset()
    # for j in xrange(200):
    #     # env.render()
    #     action = agent.action(state)  # direct action for test
    #     state, reward, done, _ = env.step(action)
    #     total_reward += reward
    #     if done:
    #         break
    pass


# ave_reward = total_reward / 100
# print 'Evaluation Average Reward:', ave_reward

if __name__ == '__main__':
    # evaluate()
    x = 5
    x /= 2
    print( x )