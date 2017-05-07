import numpy as np


def running_avg(avg, new, update_rate=0.1):
    if avg is None:
        avg = new
    else:
        avg = (1 - update_rate) * avg + update_rate * new
    return avg


def discounted_rewards(rewards, discount_factor=0.99):
    disc_rewards = np.zeros_like(rewards)
    reward = 0
    for idx in reversed(range(len(rewards))):
        reward = rewards[idx] + discount_factor * reward
        disc_rewards[idx] = reward
    return disc_rewards


class Agent(object):

    def __init__(self, network, global_network,
                 discount=0.99,
                 rollout_len=10,
                 max_steps=10**4,
                 state_fun=None):
        self.network = network
        self.global_network = global_network
        self.discount = discount
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

    def explore(self, env, sess, nb_episode=None, callback=None):
        episode = 0
        nb_step_tot = 0
        nb_update = 0
        loss_avg = None
        reward_avg = None
        value_avg = None
        if nb_episode is None:
            nb_episode = np.Inf
        action_space = np.arange(self.network.nb_action)

        while episode < nb_episode:
            episode += 1
            step = 0
            terminal = False
            reward_episode = 0
            observation = env.reset()
            state = self._get_state(observation)
            rnn_state = np.zeros([1, self.network.rnn.nb_unit],
                                 dtype=np.float32)

            while not terminal and step < self.max_steps:
                rnn_state_rollout = rnn_state
                states = []
                actions = []
                rewards = []
                values = []

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

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value)

                    state = post_state

                if terminal:
                    rewards.append(0)
                else:
                    reward = sess.run(
                        self.network.value,
                        feed_dict={self.network.state: state,
                                   self.network.rnn.init_state: rnn_state})
                    rewards.append(reward[0, 0])

                states = np.vstack(states)
                if states.ndim == 2 and states.shape[1] == 1:
                    states = states.squeeze(1)
                actions = np.stack(actions).ravel()
                rewards = np.stack(rewards).ravel()
                values = np.stack(values).ravel()

                disc_rewards = discounted_rewards(rewards, self.discount)[:-1]
                advantages = disc_rewards - values
                target_values = disc_rewards

                loss, *_ = sess.run(
                    [self.network.loss, self.network.update],
                    feed_dict={self.network.state: states,
                               self.network.rnn.init_state: rnn_state_rollout,
                               self.network.action: actions,
                               self.network.target_value: target_values,
                               self.network.advantage: advantages})
                loss_avg = running_avg(loss_avg, loss)
                value_avg = running_avg(value_avg, values.mean())

                nb_update += 1

            reward_avg = running_avg(reward_avg, reward_episode)
            if callback:
                callback(episode=episode,
                         nb_step=step,
                         nb_step_tot=nb_step_tot,
                         nb_update=nb_update,
                         reward_episode=reward_episode,
                         reward_avg=reward_avg,
                         value_avg=value_avg,
                         loss_avg=loss_avg)
