import numpy as np
from new_rnn import Controller
import gym
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

ENV_NAME = 'Pendulum-v0'
EPISODES = 1  # 100000
STEPS = 200
TEST = 100


def evaluate():
    env = gym.make(ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Controller(state_dim,action_dim,nodes=6,dt=0.05)
    agent.init_vars()

    total_reward = 0;
    for episode in xrange(EPISODES):
        observation = env.reset()
        print "episode:", episode
        # Train

        next_state = np.ones([agent.state_size, 1], dtype=np.float32)
        for step in xrange(STEPS):
            env.render()
            # print(observation.shape)
            # print(type(observation))
            # observation.reshape((3,1))
            # observation = tf.convert_to_tensor(observation)
            # tf.reshape(observation,[3,1])
            # print(type(observation))
            observation = np.reshape(observation,(3,1))
            print(observation.shape)
            action, next_state = agent.percieve(observation, next_state)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            print('step',step, 'action', action, 'observation', observation)

            if done:
                break

    print(agent.get_weights().eval())

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
    evaluate()
