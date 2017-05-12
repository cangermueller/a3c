import numpy as np
import tensorflow as tf

from .utils import running_avg


class Callback(object):

    def __init__(self, logger=print, log_sep='  '):
        self.logger = logger
        self.log_sep = log_sep

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def on_episode_start(self, episode, step=0, logs=None):
        pass

    def on_step(self, episode, step, logs=None):
        pass

    def on_update(self, episode, step, logs=None):
        pass

    def on_episode_end(self, episode, step, logs=None):
        pass


class Tensorboard(Callback):

    def __init__(self, sess, summary_writer, *args, **kwargs):
        self.sess = sess
        self.summary_writer = summary_writer

        self.episode_metrics = ['reward']
        self.update_metrics = ['loss', 'action_loss', 'value_loss',
                               'entropy_loss', 'value', 'global_norm']
        self.scalars = dict()
        for name in self.episode_metrics + self.update_metrics:
            self.scalars[name] = tf.placeholder(shape=(), dtype=tf.float32)
        self.summaries = []
        for name, op in self.scalars.items():
            self.summaries.append(tf.summary.scalar(name, op))
        self.merged_summaries = tf.summary.merge(self.summaries)

        super(Tensorboard, self).__init__(*args, **kwargs)

    def on_episode_start(self, episode, step, logs=None):
        self.avgs = dict()
        for name in self.update_metrics:
            self.avgs[name] = []

    def on_update(self, episode, step, logs=None):
        for name in self.update_metrics:
            self.avgs[name].append(logs[name])

    def on_episode_end(self, episode, step, logs=None):
        values = dict()
        for name in self.episode_metrics:
            values[name] = logs[name]
        for name, value in self.avgs.items():
            values[name] = np.mean(value)
        feed_dict = dict()
        for name, op in self.scalars.items():
            feed_dict[op] = values[name]
        summaries = self.sess.run(self.merged_summaries, feed_dict=feed_dict)
        self.summary_writer.add_summary(summaries, episode)


class Train(Callback):

    def __init__(self, avg_factor=0.1, *args, **kwargs):
        self.avg_factor = avg_factor
        self.avgs = dict()
        self.avgs_step = ['loss', 'action_loss', 'value_loss', 'entropy_loss',
                          'value', 'global_norm']
        self.avgs_episode = ['reward']
        for name in self.avgs_episode + self.avgs_step:
            self.avgs[name] = None
        self.nb_step = 0
        self.nb_update = 0
        super(Train, self).__init__(*args, **kwargs)

    def on_step(self, episode, step, logs=None):
        self.nb_step += 1

    def on_update(self, episode, step, logs=None):
        self.nb_update += 1
        for name in self.avgs_step:
            self.avgs[name] = running_avg(self.avgs[name], logs[name],
                                          self.avg_factor)

    def on_episode_end(self, episode, step, logs=None):
        for name in self.avgs_episode:
            self.avgs[name] = running_avg(self.avgs[name], logs[name],
                                          self.avg_factor)
        msg = ['episode=%d' % episode,
               'reward=%.2f' % logs['reward'],
               'steps=%d' % step,
               'steps_tot=%d' % self.nb_step,
               'updates_tot=%d' % self.nb_update,
               'reward=%.2f' % self.avgs['reward'],
               'loss=%.4f' % self.avgs['loss'],
               'action_loss=%.4f' % self.avgs['action_loss'],
               'value_loss=%.4f' % self.avgs['value_loss'],
               'entropy_loss=%.4f' % self.avgs['entropy_loss'],
               'value=%.2f' % self.avgs['value'],
               'global_norm=%.2f' % self.avgs['global_norm']]
        self.log(self.log_sep.join(msg))


class Play(Callback):

    def __init__(self, env, render_freq=1, *args, **kwargs):
        self.env = env
        self.render_freq = render_freq
        super(Play, self).__init__(*args, **kwargs)

    def on_episode_start(self, episode, step, logs=None):
        if self.render_freq:
            self.env.render()

    def on_step(self, episode, step, logs=None):
        if step % self.render_freq == 0:
            self.env.render()

    def on_episode_end(self, episode, step, logs=None):
        msg = ['episode=%d' % episode,
               'reward=%.2f' % logs['reward'],
               'steps=%d' % step]
        self.log(self.log_sep.join(msg))
