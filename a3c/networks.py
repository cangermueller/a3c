import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim


class Rnn(object):

    def __init__(self, cell_class=rnn.GRUCell, nb_unit=128):
        self.cell_class = cell_class
        self.nb_unit = nb_unit

    def __call__(self, inputs):
        self.init_state = tf.placeholder(shape=(None, self.nb_unit),
                                         dtype=tf.float32)
        self.cell = self.cell_class(self.nb_unit)
        self.outputs, self.states = rnn.static_rnn(
            tf.unpack(inputs), self.cell, initial_state=self.init_state)


class Network(object):

    def __init__(self, state, nb_action, rnn, prepro_state=None):
        self.state = state
        if prepro_state is None:
            self.prepro_state = self.state
        else:
            self.prepro_state = prepro_state
        self.nb_action = nb_action
        self.rnn = rnn
        self._build()

    def _build_stem(self):
        pass

    def _build(self):
        self._build_stem()
        self.rnn_input = tf.expand_dims(self.stem, 0)
        self.rnn(self.rnn_inputs)
        rnn_output = tf.squeeze(self.rnn.outputs, axis=0)
        self.action_dist = slim.full_connected(
            rnn_output, self.nb_action, activation_rn=tf.nn.softmax)
        self.value = slim.full_connected(
            rnn_output, 1, activation_fn=None)

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            tf.get_variable_scope().name)


class Mlp(Network):

    def __init__(self, nb_hidden=[10], *args, **kwargs):
        self.nb_hidden = nb_hidden
        super(Mlp, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self.prepro_state
        for nb_hidden in self.nb_hidden:
            layer = slim.fully_connected(layer, nb_hidden,
                                         activation_fn=tf.nn.relu)
        self.stem = layer
        return self.stem


class Cnn(Network):

    def __init__(self,
                 nb_kernel=[64, 128],
                 kernel_sizes=[3, 3],
                 pool_sizes=[2, 2],
                 nb_hidden=[512],
                 dropout=0.0,
                 *args, **kwargs):
        self.nb_kernel = nb_kernel
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.nb_hidden = nb_hidden
        self.dropout = dropout
        super(Cnn, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self.prepro_state
        for idx in range(len(self.nb_kernel)):
            layer = slim.conv2d(layer, self.nb_kernel[idx],
                                self.kernel_sizes[idx])
            layer = slim.max_pool2d(layer, self.pool_sizes[idx])
        layer = slim.flatten(layer)
        for nb_hidden in self.nb_hidden:
            layer = slim.fully_connected(layer, nb_hidden)
        if self.dropout:
            layer = slim.dropout(layer, 1 - self.dropout)
        self.stem = layer
        return self.stem
