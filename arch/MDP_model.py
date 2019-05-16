from utils.general import build_nn_module
import tensorflow as tf


class transistion_dynamic_model:
    '''
    Use neural network to predict next state given the current state
    '''

    def __init__(self, num_features, num_outputs, **kwargs):

        self.num_features = num_features
        self.num_outputs = num_outputs


        for k, item in kwargs.items():
            setattr(self, k, item)

    def _init_tf_params(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.num_features], name='inputs')
        self.targets = tf.placeholder(tf.float32, shape=[None, self.num_outputs], name='targets')

        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.global_step = tf.placeholder(tf.int32, shape=(), name='global_step')

    def _build__model(self):
        self._init_tf_params()

class reward_estimation_model:

    def __init__(self, **kwargs):
        for k, item in kwargs.items():
            setattr(self, k, item)