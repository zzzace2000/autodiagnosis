import numpy as np
import tensorflow as tf


def SE_kernel(length, x1, x2):
    x1 = tf.reshape(x1,[-1, 1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.pow(x1-x2, 2.0)/length)
    return K


def OU_kernel(length, x1, x2):
    x1 = tf.reshape(x1,[-1, 1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.abs(x1-x2)/length)
    return K


def cross_entropy_loss(labels, probs, epsilon=1e-12):
    '''
    Return the binary cross entropy loss
    labels: numpy array

    '''
    if isinstance(labels, list):
        labels = np.concatenate(labels)

    if len(labels.shape) == 1:
        labels = labels[:, None]

    probs = np.clip(probs, epsilon, 1. - epsilon)
    return - labels * np.log(probs) - (1 - labels) * np.log(1 - probs)


def ou_kernel_np(x, length):
    """ Correlation function """

    x1 = np.reshape(x, [-1, 1])  # col vec
    x2 = np.reshape(x, [1, -1])  # row vec
    K_xx = np.exp(-np.abs(x1 - x2) / length)
    return K_xx


def sigmoid(x, scale=2):
    return 1 / (1 + np.exp(-x * scale))
