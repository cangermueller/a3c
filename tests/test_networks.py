import numpy as np
from numpy import testing as npt
import tensorflow as tf

from a3c import networks

def test_cnn():
    image_shape = (16, 16, 3)
    rnn = networks.Rnn(nb_unit=32)
    state_op = tf.placeholder(shape=image_shape, dtype=tf.float32)
    cnn = networks.Cnn(state=state_op,
                       nb_action=10,
                       rnn=rnn,
                       nb_kernel=(4, 8),
                       kernel_size=(3, 3),
                       pool_sizes=(2, 2),
                       nb_hidden=(64))

    sess = tf.Session()
    for step in range(10):
        image = np.random.uniform(0, 1, image_shape)
        rnn_state = np.zeros((1, rnn.nb_unit), dtype=np.float32)
        action_dist, value, rnn_state = sess.run(
            [cnn.action_dist, cnn.value, cnn.rnn.states],
            feed_dict={state_op: image, cnn.rnn.init_state: rnn_state})
        rnn_state = rnn_state[-1]
        assert len(action_dist) == cnn.nb_action
        assert action_dist.min() == 0
        assert action_dist.max() == 0
        npt.assert_almost_equal(action_dist.sum(), 1)
