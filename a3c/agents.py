import numpy as np
import tensorflow as tf


def running_avg(avg, new, update_rate=0.1):
    if avg is None:
        avg = new
    else:
        avg = (1 - update_rate) * avg + update_rate * new
    return avg


class Agent(object):

    def __init__(self, network, global_network,
                 learning_rate=0.001,
                 discount=0.99,
                 rollout_len=10,
                 huber_loss=False,
                 max_grad_norm=None,
                 max_steps=10**4,
                 state_fun=None):
        self.network = network
        self.global_network = global_network
        self.learning_rate = learning_rate
        self.discount = discount
        self.rollout_len = rollout_len
        self.huber_loss = huber_loss
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.state_fun = state_fun

    def _get_state(self, observation):
        if self.state_fun is None:
            return observation
        else:
            return self.state_fun(observation)

    def explore(self, env, nb_episode=None, callback=None):
        episode = 0
        nb_step_tot = 0
        nb_update = 0
        reward_avg = None

        while episode < nb_episode:
            episode += 1
            observation = env.reset()
            state = self._get_state(observation)
            terminal = False
            reward_episode = 0
            rnn_state = np.zeros([1, self.network.rnn.nb_unit],
                                 dtype=np.float32)

            while not terminal and step < self.max_steps:
                step += 1
                nb_step_tot += 1

                states = []
                actions = []
                rewards = []
                values = []

                step = 0
                while step < self.rollout_len and not terminal:
                    step += 1
                    action_dist, value, rnn_state = sess.run(
                        [self.network.action_dist, self.network.value,
                         self.network.rnn.states],
                        feed_dict={self.network.state: state,
                                   self.network.rnn.state: rnn_state})
                    rnn_state = rnn_state[-1]

                    action = np.random.multinomial(1, action_dist).argmax()
                    observation, reward, terminal, info = env.step(action)
                    state = self._get_state(observation)

                    states.append(state)
                    actions.append(actions)
                    rewards.append(reward)
                    values.append(value)

                if terminal:
                    reward = rewards[-1]
                else:
                    reward = sess.run(
                        self.network.value,
                        feed_dict={self.network.state: state,
                                   self.network.rnn.state: rnn_state})
                    rnn_state = rnn_state[-1]
