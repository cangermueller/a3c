

class Callback(object):

    def __init__(self, logger=print):
        self.logger = logger
        pass

    def log(self, msg):
        if self.logger is not None:
            self.logger(msg)

    def on_episode_start(self, episode, step=0, logs=None):
        pass

    def on_step(self, episode, step, logs=None):
        pass

    def on_episode_end(self, episode, step, logs=None):
        pass


class Play(Callback):

    def __init__(self, env, render_freq=1, log_sep=' ', *args, **kwargs):
        self.env = env
        self.render_freq = render_freq
        self.log_sep = log_sep
        super(Play, self).__init__(*args, **kwargs)

    def on_episode_start(self, episode, step, logs=None):
        if self.render_freq:
            self.env.render()

    def on_step(self, episode, step, logs=None):
        if step % self.render_freq == 0:
            self.env.render()

    def on_episode_end(self, episode, step, logs=None):
        msg = ['episode=%d' % episode,
               'steps=%d' % step,
               'reward=%d' % logs['reward']]
        self.log(self.log_sep.join(msg))
