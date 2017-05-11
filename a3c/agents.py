import numpy as np

from .utils import to_list


def discounted_rewards(rewards, discount_factor=0.99):
    disc_rewards = np.zeros_like(rewards)
    reward = 0
    for idx in reversed(range(len(rewards))):
        reward = rewards[idx] + discount_factor * reward
        disc_rewards[idx] = reward
    return disc_rewards


def remove_scope(name, sep='/'):
    return sep.join(name.split(sep)[1:])


def copy_variables(src_variables, dst_variables):
    ops = []
    for src_variable, dst_variable in zip(src_variables, dst_variables):
        assert remove_scope(src_variable.name) == \
            remove_scope(dst_variable.name)
        ops.append(dst_variable.assign(src_variable.value()))
    return ops


class Agent(object):

    def __init__(self, network,
                 global_network=None,
                 gamma=0.99,
                 lambd=0.0,
                 rollout_len=10,
                 max_steps=10**4,
                 state_fun=None):
        self.network = network
        self.global_network = global_network
        self.gamma = gamma
        self.lambd = lambd
        self.rollout_len = rollout_len
        self.max_steps = max_steps
        self.state_fun = state_fun

    def _get_state(self, observation):
        if self.state_fun is None:
            state = observation
        else:
            state = self.state_fun(observation)
        state = np.expand_dims(state, 0)
        return state

    def train(self, env, sess, nb_episode=None, stop_fun=None, callbacks=None):
        if nb_episode is None:
            nb_episode = np.Inf
        callbacks = to_list(callbacks)
        episode = 0
        nb_step_tot = 0
        nb_update = 0
        action_space = np.arange(self.network.nb_action)

        if self.global_network is None:
            copy_from_global = None
            apply_variables = self.network.trainable_variables
        else:
            copy_from_global = copy_variables(
                self.global_network.trainable_variables,
                self.network.trainable_variables)
            apply_variables = self.global_network.trainable_variables
        apply_gradients = self.network.optimizer.apply_gradients(
            zip(self.network.gradients, apply_variables))

        for episode in range(1, nb_episode + 1):
            step = 0
            terminal = False
            reward_episode = 0
            observation = env.reset()
            state = self._get_state(observation)
            rnn_state = np.zeros([1, self.network.rnn.nb_unit],
                                 dtype=np.float32)
            if callbacks is not None:
                logs = {'state': state}
                for callback in callbacks:
                    callback.on_episode_start(episode, step, logs=logs)

            while not terminal and step < self.max_steps:
                if stop_fun is not None and stop_fun(episode, step):
                    return
                rnn_state_rollout = rnn_state
                states = []
                actions = []
                rewards = []
                values = []
                if copy_from_global is not None:
                    # Copy global parameters
                    sess.run(copy_from_global)

                while len(states) < self.rollout_len and not terminal:
                    step += 1
                    nb_step_tot += 1
                    action_dist, value, rnn_state = sess.run(
                        [self.network.action_dist, self.network.value,
                         self.network.rnn.states],
                        feed_dict={self.network.state: state,
                                   self.network.rnn.init_state: rnn_state})
                    action = np.random.choice(action_space, p=action_dist[0])
                    observation, reward, terminal, info = env.step(action)
                    reward_episode += reward
                    post_state = self._get_state(observation)

                    if callbacks is not None:
                        logs = {'state': state,
                                'action': action,
                                'reward': reward,
                                'post_state': post_state}
                        for callback in callbacks:
                            callback.on_step(episode, step, logs=logs)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value.squeeze())

                    state = post_state

                if terminal:
                    rewards.append(0)
                    values.append(0)
                else:
                    value = sess.run(
                        self.network.value,
                        feed_dict={self.network.state: state,
                                   self.network.rnn.init_state: rnn_state})
                    value = value.squeeze()
                    rewards.append(value)
                    values.append(value)

                states = np.vstack(states)
                if states.ndim == 2 and states.shape[1] == 1:
                    states = states.squeeze(1)
                actions = np.stack(actions).ravel()
                rewards = np.stack(rewards).ravel()
                values = np.stack(values).ravel()

                disc_rewards = discounted_rewards(rewards, self.gamma)[:-1]
                target_values = disc_rewards
                # Generalizes advantage estimation
                td_errors = rewards[:-1] + self.gamma * values[1:] \
                    - values[:-1]
                advantages = discounted_rewards(
                    td_errors, self.gamma * self.lambd)

                loss, action_loss, value_loss, entropy_loss, global_norm, *_ = \
                    sess.run(
                        [self.network.loss,
                         self.network.action_loss,
                         self.network.value_loss,
                         self.network.entropy_loss,
                         self.network.global_norm,
                         apply_gradients],
                        feed_dict={self.network.state: states,
                                   self.network.rnn.init_state:
                                   rnn_state_rollout,
                                   self.network.action: actions,
                                   self.network.target_value: target_values,
                                   self.network.advantage: advantages}
                    )
                nb_update += 1
                logs = {'loss': loss,
                        'value_loss': value_loss,
                        'action_loss': action_loss,
                        'entropy_loss': entropy_loss,
                        'value': values.mean(),
                        'global_norm': global_norm}
                if callbacks is not None:
                    for callback in callbacks:
                        callback.on_update(episode, step, logs=logs)

            # Episode end
            logs = {'reward': reward_episode,
                    'terminal': terminal}
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_episode_end(episode, step, logs=logs)

    def play(self, env, sess, nb_episode=None, stop_fun=None, callbacks=None):
        if nb_episode is None:
            nb_episode = np.Inf
        callbacks = to_list(callbacks)
        action_space = np.arange(self.network.nb_action)

        for episode in range(1, nb_episode + 1):
            step = 0
            terminal = False
            reward_episode = 0
            observation = env.reset()
            state = self._get_state(observation)
            rnn_state = np.zeros([1, self.network.rnn.nb_unit],
                                 dtype=np.float32)
            if callbacks is not None:
                logs = {'state': state}
                for callback in callbacks:
                    callback.on_episode_start(episode, step, logs=logs)

            while not terminal and step < self.max_steps:
                step += 1
                action_dist, rnn_state = sess.run(
                    [self.network.action_dist,
                     self.network.rnn.states],
                    feed_dict={self.network.state: state,
                               self.network.rnn.init_state: rnn_state})
                action = np.random.choice(action_space, p=action_dist[0])
                observation, reward, terminal, info = env.step(action)
                reward_episode += reward
                post_state = self._get_state(observation)

                if callbacks is not None:
                    logs = {'state': state,
                            'action': action,
                            'reward': reward,
                            'post_state': post_state}
                    for callback in callbacks:
                        callback.on_step(episode, step, logs=logs)

                state = post_state

            if callbacks is not None:
                logs = {'reward': reward_episode,
                        'terminal': terminal}
                for callback in callbacks:
                    callback.on_episode_end(episode, step, logs=logs)
