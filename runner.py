'''
The Runner class, is used to generate an agent (RNN controller),
and a simulator (NARX neural net)
'''

# Make the agent
#
from rnn import FullyConnectedRNN
obs_dim = env.observation_space.shape[0]  # Input to controller (observ.)
action_dim = env.action_space.shape[0]  # Output from controller (action)
nodes = 10  # Unconnected nodes in network in the
dt = 0.05  # dt for the environment, found in environment source code
dt = 1
agent = FullyConnectedRNN()

# Test the agent in the "real world"
#   and collect sensory information, motor commands & reward (may have to
# just make a reward function



# Make the simulator
#    Make NARX neural net



# connect to real world
#


