#!/usr/bin/env python -u

from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import os
import random
import sys
import threading
from time import sleep

import argparse
import logging
import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf

from a3c import networks as net
from a3c import callbacks as cbk
from a3c.agents import Agent
from a3c.utils import make_dir, rgb2y


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
            description='Advantage Asynchronous Actor Critic (A3C)')
        p.add_argument(
            '--env',
            help='Environment',
            default='FrozenLake-v0')
        p.add_argument(
            '--nb_train',
            help='Number of episodes',
            type=int,
            default=100)
        p.add_argument(
            '--nb_play',
            help='Number of episodes to play',
            type=int,
            default=0)

        # IO options
        p.add_argument(
            '--in_checkpoint',
            help='Input checkpoint path')
        p.add_argument(
            '--out_checkpoint',
            help='Output checkpoint path')
        p.add_argument(
            '--save_freq',
            help='Frequency in epochs to create checkpoint',
            type=int,
            default=1000)
        p.add_argument(
            '--monitor',
            help='Gym monitor output directory')
        p.add_argument(
            '--tensorboard',
            help='Tensorboard output directory')

        # Learning parameters
        p.add_argument(
            '--nb_agent',
            help='Number of agents',
            type=int,
            default=1)
        p.add_argument(
            '--learning_rate',
            help='Learning rate',
            type=float,
            default=0.0001)
        p.add_argument(
            '--gamma',
            help='Gamme discount factor',
            type=float,
            default=0.99)
        p.add_argument(
            '--lambd',
            help='Lambda discount factor',
            type=float,
            default=0.0)
        p.add_argument(
            '--rollout_len',
            help='Rollout length',
            type=int,
            default=5)
        p.add_argument(
            '--max_grad_norm',
            help='Maximum gradient norm',
            type=float)
        p.add_argument(
            '--entropy_weight',
            help='Weight of entropy loss',
            type=float,
            default=0.0)

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
            default=[2, 2])
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

    def save_graph(self):
        out_path = self.opts.out_checkpoint
        if not out_path:
            return
        if not os.path.isdir(out_path):
            out_dir = os.path.dirname(out_path)
            make_dir(out_dir)

        self.log.info('Saving graph to %s ...' % out_path)
        self.saver.save(self.sess, out_path)

    def build_mlp(self, *args, **kwargs):
        opts = self.opts
        network = net.Mlp(nb_hidden=opts.nb_hidden,
                          dropout=opts.dropout,
                          *args, **kwargs)
        return network

    def build_cnn(self, *args, **kwargs):
        opts = self.opts
        network = net.Cnn(nb_hidden=opts.nb_hidden,
                          nb_kernel=opts.nb_kernel,
                          kernel_sizes=opts.kernel_sizes,
                          pool_sizes=opts.pool_sizes,
                          dropout=opts.dropout,
                          *args, **kwargs)
        return network

    def train(self):
        opts = self.opts
        log = self.log
        coordinator = tf.train.Coordinator()

        def stop_fun(*args, **kwargs):
            coordinator.should_stop()

        threads = []
        idx = 0
        for name, agent in self.agents.items():
            log.info('Starting thread for %s ...' % name)
            idx += 1
            env = gym.make(opts.env)
            env.seed(opts.seed + idx)
            callbacks = None
            writer = None
            if idx == 1:
                callbacks = []
                callbacks.append(cbk.Train())
                if opts.monitor:
                    make_dir(opts.monitor)
                    env = Monitor(env, opts.monitor, force=True)
                if opts.tensorboard:
                    make_dir(opts.tensorboard)
                    writer = tf.summary.FileWriter(opts.tensorboard)
                    writer.add_graph(self.graph)
                    callbacks.append(cbk.Tensorboard(self.sess, writer))

            def thread_fun(graph=tf.get_default_graph()):
                with graph.as_default():
                    agent.train(env, self.sess,
                                nb_episode=opts.nb_train,
                                stop_fun=stop_fun,
                                callbacks=callbacks)
                    writer.close()

            thread = threading.Thread(target=thread_fun)
            thread.start()
            sleep(0.5)
            threads.append(thread)

        coordinator.join(threads)
        self.save_graph()

    def play(self):
        callback = cbk.Play(self.env)
        self.agents['agent01'].play(self.env, self.sess,
                                    nb_episode=self.opts.nb_play,
                                    callbacks=callback)

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
        self.env = gym.make(opts.env)

        # Set seed
        if opts.seed is not None:
            np.random.seed(opts.seed)
            random.seed(opts.seed)
            tf.set_random_seed(opts.seed)

        # Setup networks
        state_fun = None
        network_fun = self.build_mlp
        if opts.env == 'Pong-v0':
            state_shape = [80, 80, 1]
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state
            network_fun = self.build_cnn
            state_fun = pong_state_fun
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            state_shape = None
            state = tf.placeholder(tf.int32, [None], name='state')
            prepro_state = tf.one_hot(state, self.env.observation_space.n)
        else:
            state_shape = list(self.env.observation_space.shape)
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state

        network_names = ['global']
        for idx in range(opts.nb_agent):
            network_names.append('agent%02d' % (idx + 1))
        self.networks = OrderedDict()
        optimizer = tf.train.AdamOptimizer(opts.learning_rate)
        for name in network_names:
            log.info('Building %s ...' % name)
            with tf.variable_scope(name):
                rnn = net.Rnn(nb_unit=opts.nb_rnn_unit)
                network = network_fun(state=state,
                                      prepro_state=prepro_state,
                                      nb_action=self.env.action_space.n,
                                      rnn=rnn,
                                      optimizer=optimizer,
                                      max_grad_norm=opts.max_grad_norm,
                                      entropy_weight=opts.entropy_weight)
                self.networks[name] = network

        log.info('Number of network parameters: %d' %
                 count_params(self.networks['global'].trainable_variables))

        # Setup agents
        self.agents = OrderedDict()
        for name, network in self.networks.items():
            if name == 'global':
                continue
            self.agents[name] = Agent(network=network,
                                      global_network=self.networks['global'],
                                      gamma=opts.gamma,
                                      lambd=opts.lambd,
                                      rollout_len=opts.rollout_len,
                                      state_fun=state_fun)

        self.graph = tf.get_default_graph()
        self.sess = tf.Session()

        # Load or initialize network variables
        self.saver = tf.train.Saver()
        if opts.in_checkpoint:
            log.info('Restoring graph from %s ...' % opts.in_checkpoint)
            self.saver.restore(self.sess, opts.in_checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

        if opts.nb_train:
            self.train()

        if opts.nb_play:
            self.play()
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
