import numpy as np
import tensorflow as tf

from a3c import networks


def test_cnn():
    image_shape = [16, 16, 3]
    rnn = networks.Rnn(nb_unit=32)
    state_op = tf.placeholder(shape=[None] + image_shape, dtype=tf.float32)
    optimizer = tf.train.AdamOptimizer(0.1)
    cnn = networks.Cnn(state=state_op,
                       nb_action=10,
                       rnn=rnn,
                       optimizer=optimizer,
                       max_grad_norm=5,
                       nb_kernel=(4, 8),
                       kernel_sizes=(3, 3),
                       pool_sizes=(2, 2),
                       nb_hidden=(64))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for nb_step in range(1, 10):
        image = np.random.uniform(0, 1, [nb_step] + image_shape)
        rnn_state = np.zeros((1, rnn.nb_unit), dtype=np.float32)
        action_dist, value, rnn_states = sess.run(
            [cnn.action_dist, cnn.value, cnn.rnn.states],
            feed_dict={state_op: image, cnn.rnn.init_state: rnn_state})

        assert rnn_states.shape == (1, rnn.nb_unit)
        assert action_dist.shape == (nb_step, cnn.nb_action)
        assert action_dist.min() >= 0
        assert action_dist.max() <= 1
        assert np.all(np.abs(action_dist.sum(axis=1) - 1) < 1e-4)
        assert value.shape == (nb_step, 1)

        actions = np.random.randint(0, cnn.nb_action, nb_step)
        advantages = np.random.normal(0, 1, nb_step)
        target_values = np.random.normal(0, 1, nb_step)

        loss, entropy, global_norm, *_ = sess.run(
            [cnn.loss, cnn.entropy, cnn.global_gradient_norm, cnn.update],
            feed_dict={cnn.state: image,
                       cnn.rnn.init_state: rnn_state,
                       cnn.action: actions,
                       cnn.advantage: advantages,
                       cnn.target_value: target_values})
        assert entropy >= 0
        assert global_norm >= 0
