import tensorflow as tf

def simple_rnn(inputs, state):
    """
    Inputs added to the first obs_space_dim inputs


    :param inputs:
    :param state:
    :return:
    """
    with tf.name_scope('rnn'):
        Wx_b = self.weights * state + self.bias

        Wx_b_I = Wx_b.assign(Wx_b[:self.obs_space_dim] + inputs)

        next_state = state + self.dt * (-state + tf.tanh(Wx_b_I))

        output = next_state[self.act_space_dim:]

        return output, next_state