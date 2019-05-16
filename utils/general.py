import logging
import os
import time

import numpy as np
import tensorflow as tf


def to_one_hot(inds, num_actions, zero_inds=None):
    """
    Return a one hot encoding given a numpy array inds indicating where it is hot
    """
    n_s = len(inds)
    coding = np.zeros((n_s, num_actions), dtype=int)
    coding[np.arange(n_s), inds] = 1
    if zero_inds is not None:
        coding[zero_inds] = 0
    return coding


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        print('Start %s...' % self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Finish {} in {} seconds'.format(self.name, time.time() - self.start_time))

def log(logfile=None, str=None, mode='a+'):
    """ Log a string in a file """
    if logfile is not None:
        with open(logfile, mode) as f:
            f.write(str+'\n')
    print(str)

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log

def get_logger2(logfile):
    def log(str=None, mode='a+'):
        """ Log a string in a file """
        if logfile is not None:
            with open(logfile, mode) as f:
                print(str, file=f)
        print(str)
    return log

def save_config(fname, flagdict):
    """ Save configuration """
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])

    with open(fname, 'a') as f:
        f.write(s)


def make_new_log_dir(parent_dir_name, prefix='exp'):

    num_dir = len([x[0] for x in os.walk("../exps")]) - 1

    new = parent_dir_name + '/' + prefix + str(num_dir + 1) + '/'
    while os.path.isdir(new):
        num_dir += 1
        new = parent_dir_name + '/' + prefix + str(num_dir + 1) + '/'

    os.makedirs(new)

    return new


def build_nn_module(input, num_layers, num_hidden_units, reg_constant, keep_prob, is_training, dropout_on=False):
    """ Build a neural network layer consisting of 1 fc, 1 bn and  1 drop out component """

    for i in range(num_layers):
        fc = tf.contrib.layers.fully_connected(input, num_hidden_units, activation_fn=tf.nn.relu,
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
        bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=is_training)
        input = tf.contrib.layers.dropout(bn, keep_prob=keep_prob, is_training=is_training) # todo: use dropout_on
    return input


def output_csv(the_path, data_dict):
    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        if not is_file_exists:
            print(','.join([str(k) for k in data_dict.keys()]), file=op)
        print(','.join([str(v) for v in data_dict.values()]), file=op)