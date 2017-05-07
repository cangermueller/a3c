#!/usr/bin/env python -u

from __future__ import division
from __future__ import print_function

import os
import random
import sys

import argparse
import logging
import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf

from a3c import agents
from a3c import networks as nets
from a3c.utils import rgb2y


def pong_state_fun(image):
    image = image[35:195]
    image = image[::2, ::2]
    image = image.astype(np.float32) / 256
    image = rgb2y(image) - 0.5
    image = np.expand_dims(image, axis=2)
    return image


def count_params(variables):
    nb_param = 0
    for variable in variables:
        shape = variable.get_shape().as_list()
        nb_param += np.prod(shape)
    return nb_param


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Asynchronous Actor Critic RL')
        p.add_argument(
            '--env',
            help='Environment',
            default='FrozenLake-v0')

        # IO options
        p.add_argument(
            '-i', '--in_checkpoint',
            help='Input checkpoint path')
        p.add_argument(
            '-o', '--out_checkpoint',
            help='Output checkpoint path')
        p.add_argument(
            '--save_freq',
            help='Frequency in epochs to create checkpoint',
            type=int,
            default=1000)
        p.add_argument(
            '--monitor',
            help='Output directory of gym monitor')
        p.add_argument(
            '--nb_episode',
            help='Number of episodes',
            type=int,
            default=100)
        p.add_argument(
            '--nb_play',
            help='Number of episodes to play',
            type=int,
            default=0)

        # Learning parameters
        p.add_argument(
            '--learning_rate',
            help='Learning rate',
            type=float,
            default=0.0001)
        p.add_argument(
            '--discount',
            help='Reward discount factors',
            type=float,
            default=0.99)
        p.add_argument(
            '--rollout_len',
            help='Rollout length',
            type=int,
            default=5)
        p.add_argument(
            '--max_grad_norm',
            help='Maximum gradient norm',
            type=float)

        # Network architecture
        p.add_argument(
            '--nb_rnn_unit',
            help='Number of units in recurrent layer',
            type=int,
            default=256)
        p.add_argument(
            '--nb_hidden',
            help='Number of units in FC layer',
            type=int,
            nargs='+',
            default=[10])
        p.add_argument(
            '--nb_kernel',
            help='Number of kernels in CNN',
            type=int,
            nargs='+',
            default=[32, 64])
        p.add_argument(
            '--kernel_sizes',
            help='Kernels sizes in CNN',
            type=int,
            nargs='+',
            default=[3, 3])
        p.add_argument(
            '--pool_sizes',
            help='Pooling sizes in CNN',
            type=int,
            nargs='+',
            default=[32, 64])
        p.add_argument(
            '--dropout',
            help='Dropout rate',
            type=float,
            default=0.0)

        # Misc
        p.add_argument(
            '--seed',
            help='Seed of random number generator',
            type=int,
            default=0)
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')

        return p

    def callback(self, episode, nb_step, nb_step_tot, nb_update,
                 reward_episode, reward_avg, value_avg, loss_avg,
                 *args, **kwargs):

        if episode % self.opts.save_freq == 0:
            self.save_graph()

        def format_na(x, spec):
            if x is None:
                return 'NA'
            else:
                return '{:{spec}}'.format(x, spec=spec)

        tmp = ['episode=%d' % episode,
               'steps=%d' % nb_step,
               'steps_tot=%d' % nb_step_tot,
               'updates=%d' % nb_update,
               'r_epi=%.2f' % reward_episode,
               'r_avg=%s' % format_na(reward_avg, '.2f'),
               'v_avg=%s' % format_na(value_avg, '.2f'),
               'l_avg=%s' % format_na(loss_avg, '.4f')]
        tmp = '  '.join(tmp)
        print(tmp)

    def save_graph(self):
        out_path = self.opts.out_checkpoint
        if not out_path:
            return
        if not os.path.isdir(out_path):
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

        self.log.info('Saving graph to %s ...' % out_path)
        self.saver.save(self.sess, out_path)

    def build_mlp(self, *args, **kwargs):
        opts = self.opts
        network = nets.Mlp(nb_hidden=opts.nb_hidden,
                           dropout=opts.dropout,
                           *args, **kwargs)
        return network

    def build_cnn(self, *args, **kwargs):
        opts = self.opts
        network = nets.Cnn(nb_hidden=opts.nb_hidden,
                           nb_kernel=opts.nb_kernel,
                           kernel_sizes=opts.kernel_sizes,
                           pool_sizes=opts.pool_sizes,
                           dropout=opts.dropout,
                           *args, **kwargs)
        return network

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.opts = opts
        self.log = log

        # Build environment
        env = gym.make(opts.env)
        if opts.monitor:
            os.makedirs(opts.monitor, exist_ok=True)
            env = Monitor(env, opts.monitor, force=True)

        # Set seed
        if opts.seed is not None:
            np.random.seed(opts.seed)
            random.seed(opts.seed)
            tf.set_random_seed(opts.seed)
            env.seed(opts.seed)

        # Setup networks
        state_fun = None
        network_fun = self.build_mlp
        if opts.env == 'Pong-v0':
            state_shape = [80, 80, opts.stack_size]
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state
            network_fun = self.build_cnn
            state_fun = pong_state_fun
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            state_shape = None
            state = tf.placeholder(tf.int32, [None], name='state')
            prepro_state = tf.one_hot(state, env.observation_space.n)
        else:
            state_shape = list(env.observation_space.shape)
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state

        networks = []
        for name in ['actor']:
            log.info('Building %s ...' % name)
            with tf.variable_scope(name):
                rnn = nets.Rnn(nb_unit=opts.nb_rnn_unit)
                optimizer = tf.train.AdamOptimizer(opts.learning_rate)
                networks.append(
                    network_fun(state=state,
                                prepro_state=prepro_state,
                                nb_action=env.action_space.n,
                                rnn=rnn,
                                optimizer=optimizer,
                                max_grad_norm=opts.max_grad_norm))
        network = networks[0]
        log.info('Number of network parameters: %d' %
                 count_params(network.trainable_variables))

        # Setup agent
        self.sess = tf.Session()
        agent = agents.Agent(network=network,
                             global_network=network,
                             discount=opts.discount,
                             rollout_len=opts.rollout_len,
                             state_fun=state_fun)

        # Load or initialize network variables
        self.saver = tf.train.Saver()
        if opts.in_checkpoint:
            log.info('Restoring graph from %s ...' % opts.in_checkpoint)
            self.saver.restore(self.sess, opts.in_checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

        # Explore
        if opts.nb_episode:
            agent.explore(env, self.sess,
                          nb_episode=opts.nb_episode,
                          callback=self.callback)

        # Play
        if opts.nb_play:
            agent.play(env, opts.nb_play)

        self.save_graph()
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
