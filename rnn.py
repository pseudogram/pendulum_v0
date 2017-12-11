import tensorflow as tf
import numpy as np
from pprint import pprint



# If using seed to replicate results.
# np.random.seed(22)

dt = 0.05  # time between each time step in the simulation.
num_inputs = 3  # Number of Values output by the simulation
num_outputs = 1  # OpenAI env takes a single controller input, thus the controller network must output only one.
num_nodes = num_inputs + 1  # Number of nodes in the network (must be >=nump_inputs)
num_epochs = 10

# Randomly initialize state of nodes?
initial_state = np.random.randn(num_nodes,1)  #  tf.zeros([num_nodes,1],
# dtype=tf.float32)






"""
A class used to create recurrent Neural Networks.
"""
class Other():
    def __init__( self, env ):
        self.name = 'DDPG'  # name for uploading results
        self.environment = env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        # Initialize time step
        self.time_step = 0
        # initialize replay buffer
        self.replay_buffer = deque()
        # initialize networks
        self.create_networks_and_training_method(state_dim, action_dim)

        # An Interactive session is used to run the graph.
        # The interactive session is separate 
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print
            "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print
            "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('~/logs',
                                                    graph=self.sess.graph)



def create_network( set_weights=None, set_bias=None ):
    """

    Args:
        set_weights - the weights for the RNN dimensions of (num_nodes+num_outputs,num_nodes)

    Returns:
        input - placeholder in the network to input values
        output - output
        update_state -
    """
    # _______________________________________
    #               Setting shit
    # _______________________________________
    # Randomly set the wieghts when initializing else set them to whats been given.
    if (not set_weights): set_weights = np.random.randn(num_nodes + num_outputs,
                                                        num_nodes) * 0.1
    print set_weights.shape
    print('Setting weights automatically')
    # if (not set_bias):
    print('Setting bias automatically')
    set_bias = np.zeros([num_nodes + num_outputs, 1], dtype=np.float32)

    # _______________________________________
    #               Variables
    # _______________________________________

    # Each input variable feeds into one node
    _input = tf.placeholder(tf.float32, [num_inputs,
                                         1])  # Dim 1 = 1 as this is a dynamic task that only takes one input at a time
    _input_buffer = tf.zeros([num_nodes - num_inputs, 1],
                             dtype=tf.float32)  # used so add the input to the state in one step.

    # Concatenate the inputs
    I = tf.concat([_input, _input_buffer], 0)
    print(type(set_weights))
    # Create the weights for the RNN
    W = tf.Variable(set_weights, dtype=tf.float32)  # could be a constant...
    b = tf.Variable(set_bias, dtype=tf.float32)  # could also be a constant...

    state = tf.Variable(initial_state, dtype=tf.float32)

    # _______________________________________
    #               Operations
    # _______________________________________
    Wx = tf.matmul(W, state)
    Wx_b = Wx + b
    recurrence, output = tf.split(Wx_b, [num_nodes, num_outputs], 0)

    output = tf.tanh(output) * 2

    next_state = state + dt * (-state + tf.tanh(recurrence + I))

    update_state = state.assign(next_state)

    return _input, output, W, state


_input, output, weights, update_state = create_network()

_input2, output2, weights2, update_state2 = create_network()

print("Output1")
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #     print(I.eval(session=sess))


    out_list = list()
    #     _in = np.random.randn(num_inputs,1)
    _in = np.zeros([num_inputs, 1], dtype=np.float32)
    for epoch in range(num_epochs):
        _, out = sess.run([update_state, output], feed_dict={_input: _in})
        out_list.append(out)
pprint(out_list)

print()
print("Output2")
with tf.Session() as sess2:
    init2 = tf.global_variables_initializer()
    sess2.run(init2)

    #     print(I.eval(session=sess))


    out_list2 = list()
    #     _in = np.random.randn(num_inputs,1)
    _in2 = np.zeros([num_inputs, 1], dtype=np.float32)
    for epoch in range(num_epochs):
        _, out2 = sess2.run([update_state2, output2], feed_dict={_input2: _in2})
        out_list2.append(out2)

pprint(out_list2)

print()
print("Output1.2")
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #     print(I.eval(session=sess))


    out_list = list()
    #     _in = np.random.randn(num_inputs,1)
    _in = np.zeros([num_inputs, 1], dtype=np.float32)
    for epoch in range(num_epochs):
        _, out = sess.run([update_state, output], feed_dict={_input: _in})
        out_list.append(out)

pprint(out_list)
print('hello')
# pprint(out_list2)

print("Output1.2")
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #     print(I.eval(session=sess))


    out_list = list()
    #     _in = np.random.randn(num_inputs,1)
    _in = np.zeros([num_inputs, 1], dtype=np.float32)
    for epoch in range(num_epochs):
        _, out = sess.run([update_state, output], feed_dict={_input: _in})
        out_list.append(out)

pprint(type(out))
print('hello')
# pprint(out_list2)
