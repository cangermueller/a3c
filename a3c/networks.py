import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

from .utils import to_list


class Rnn(object):

    def __init__(self, cell_class=rnn.GRUCell, nb_unit=128):
        self.cell_class = cell_class
        self.nb_unit = nb_unit

    def __call__(self, inputs):
        self.init_state = tf.placeholder(shape=(None, self.nb_unit),
                                         dtype=tf.float32)
        self.cell = self.cell_class(self.nb_unit)
        self.outputs, self.states = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.init_state)


class Network(object):

    def __init__(self, state, nb_action, rnn, optimizer,
                 prepro_state=None,
                 max_grad_norm=5,
                 entropy_weight=0.001,
                 ):
        self.state = state
        if prepro_state is None:
            self.prepro_state = self.state
        else:
            self.prepro_state = prepro_state
        self.nb_action = nb_action
        self.rnn = rnn
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.entropy_weight = entropy_weight

        self._build()

    def _build_stem(self):
        pass

    def _build_objective(self):
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_value = tf.placeholder(shape=[None], dtype=tf.float32)

        action_onehot = tf.one_hot(self.action, self.nb_action)
        self.log_action_dist = tf.log(self.action_dist + 1e-6)
        self.log_action_selected = tf.reduce_sum(
            self.log_action_dist * action_onehot, axis=1)
        # log(pi(q|s)) * (R - V(s))
        self.action_loss = -tf.reduce_mean(
            self.log_action_selected * self.advantage)
        self.entropy = -tf.reduce_mean(self.action_dist * self.log_action_dist)
        self.entropy_loss = -self.entropy * self.entropy_weight
        # (R - V(s))**2
        self.value_loss = tf.reduce_mean(
            tf.squared_difference(self.target_value, tf.squeeze(self.value)))
        self.loss = self.action_loss + self.value_loss + self.entropy_loss
        self.gradients = tf.gradients(
            self.loss, self.trainable_variables)
        if self.max_grad_norm:
            self.gradients, self.global_norm = \
                tf.clip_by_global_norm(self.gradients, self.max_grad_norm)
        else:
            self.global_norm = tf.constant(-1)
        self.update = self.optimizer.apply_gradients(
            zip(self.gradients, self.trainable_variables))

    def _build(self):
        self._build_stem()
        # Batch size of 1
        self.rnn_inputs = tf.expand_dims(self.stem, axis=0)
        self.rnn(self.rnn_inputs)
        self.rnn_outputs = tf.squeeze(self.rnn.outputs, axis=0)
        self.action_dist = slim.fully_connected(
            self.rnn_outputs, self.nb_action, activation_fn=tf.nn.softmax)
        self.value = slim.fully_connected(
            self.rnn_outputs, 1, activation_fn=None)

        self.trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            tf.get_variable_scope().name)

        if self.optimizer is not None:
            self._build_objective()


class Mlp(Network):

    def __init__(self, nb_hidden=[10], dropout=0.0, *args, **kwargs):
        self.nb_hidden = nb_hidden
        self.dropout = dropout
        super(Mlp, self).__init__(*args, **kwargs)

    def _build_stem(self):
        layer = self.prepro_state
        for nb_hidden in self.nb_hidden:
            layer = slim.fully_connected(layer, nb_hidden,
                                         activation_fn=tf.nn.relu)
        if self.dropout:
            layer = slim.dropout(layer, 1 - self.dropout)
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
        self.nb_kernel = to_list(nb_kernel)
        self.kernel_sizes = to_list(kernel_sizes)
        self.pool_sizes = to_list(pool_sizes)
        self.nb_hidden = to_list(nb_hidden)
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
