import os
import tensorflow as tf
import sys
import numpy as np
import pickle
import json


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class MIMIC_cache_discretized_exp(object):
    def __init__(self, cache_dir, action_dim=None):
        self.action_dim = action_dim
        self.cache_dir = cache_dir
        self.mimic_exp = None

        self._read_dataset()

    def _get_features(self):
        return {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'gain_per_action': tf.FixedLenFeature([], tf.string),
            'action': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
        }

    def _read_dataset(self):
        # A vector of filenames.
        self.filenames = tf.placeholder(tf.string, shape=[1])
        self.batch_size = tf.placeholder(tf.int64, shape=[])
        self.shuffle_buffer_size = tf.placeholder(tf.int64, shape=[])

        self.dataset = tf.data.TFRecordDataset(self.filenames)

        self.dataset = self.dataset.map(self._parse_function)  # Parse the record into tensors.
        self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        self.dataset = self.dataset.repeat(count=1)  # Repeat the input indefinitely.
        self.dataset = self.dataset.batch(self.batch_size)

        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        if self.action_dim is not None:
            self.next_element['actions'] = tf.one_hot(self.next_element['actions'], self.action_dim)
            self.next_element['actions'] = tf.cast(self.next_element['actions'], tf.int32)

    def _parse_function(self, example_proto):
        parsed_features = tf.parse_single_example(example_proto, self._get_features())

        # Convert the image data from string back to the numbers
        cur_state = tf.decode_raw(parsed_features['cur_state'], tf.float32)
        next_state = tf.decode_raw(parsed_features['next_state'], tf.float32)

        result = {
            'cur_state': cur_state,
            'next_state': next_state,
            'labels': tf.cast(parsed_features['label'], tf.int32),
            'patient_ind': tf.cast(parsed_features['patient_inds'], tf.int32),
            'the_step': tf.cast(parsed_features['the_steps'], tf.int32),
            'total_steps': tf.cast(parsed_features['total_steps'], tf.int32)
        }

        self._handle_specific_feature(result, parsed_features)
        return result

    def _handle_specific_feature(self, result, parsed_features):
        result['actions'] = tf.cast(parsed_features['action'], tf.int32)

        gain = tf.decode_raw(parsed_features['gain_per_action'], tf.float32)
        result['gain'] = gain[0]

    def gen_train_experience(self, sess, batch_size, shuffle=True):
        ''' Only generate experiences from those patients who died. '''

        # Initialize `iterator` with training data.
        filenames = [os.path.join(self.cache_dir, 'train.tfrecords')]
        return self._gen_experience(sess, filenames, batch_size, shuffle_buffer_size=10000 if shuffle else 1)

    def gen_val_experience(self, sess, batch_size, shuffle=False):
        ''' Only generate experiences from those patients who died. '''

        # Initialize `iterator` with val data.
        filenames = [os.path.join(self.cache_dir, 'val.tfrecords')]
        return self._gen_experience(sess, filenames, batch_size, shuffle_buffer_size=10000 if shuffle else 1)

    def gen_test_experience(self, sess, batch_size, shuffle=False):
        ''' Only generate experiences from those patients who died. '''

        # Initialize `iterator` with val data.
        filenames = [os.path.join(self.cache_dir, 'test.tfrecords')]
        return self._gen_experience(sess, filenames, batch_size, shuffle_buffer_size=10000 if shuffle else 1)

    def gen_testing_experience(self, sess, batch_size, shuffle=False):
        filenames = [os.path.join(self.cache_dir, 'testing.tfrecords')]
        return self._gen_experience(sess, filenames, batch_size, shuffle_buffer_size=10000 if shuffle else 1)

    def gen_experience(self, filename, sess, batch_size, shuffle=False):
        filenames = [os.path.join(self.cache_dir, '%s.tfrecords' % filename)]
        return self._gen_experience(sess, filenames, batch_size, shuffle_buffer_size=10000 if shuffle else 1)

    def get_per_time_patient_data_loader(self, sess, mode, batch_size):
        if self.mimic_exp is None:
            self.mimic_exp = self._init_mimic_exp(sess)

        self.mimic_exp.gen_pat_data_loader(mode=mode, batch_size=batch_size)

        return self.mimic_exp.gen_experience('%s_per_time_env' % mode)

    def _init_mimic_exp(self, sess):
        hyperparams = json.load(open(os.path.join(self.cache_dir, 'train_hyperparams.json')))

        # load rnn
        from arch.RNN import ManyToOneRNN
        hyperparams['rnn'] = ManyToOneRNN.load_saved_mgp_rnn(sess, hyperparams['mgp_rnn_dir'])

        from database.MIMIC_discretized_exp import MIMIC_per_time_env_exp_rnn
        self.mimic_exp = MIMIC_per_time_env_exp_rnn(**hyperparams)

    def _gen_experience(self, sess, filenames, batch_size, shuffle_buffer_size):
        # Compute for 1 epochs.
        for _ in range(1):
            # Initialize
            sess.run(self.iterator.initializer,
                     feed_dict={self.filenames: filenames,
                                self.batch_size: batch_size,
                                self.shuffle_buffer_size: shuffle_buffer_size})
            while True:
                try:
                    yield sess.run(self.next_element)
                except tf.errors.OutOfRangeError:
                    break  # End of epoch


class MIMIC_cache_discretized_exp_joint(MIMIC_cache_discretized_exp):
    def _get_features(self):
        features = super(MIMIC_cache_discretized_exp_joint, self)._get_features()
        # Rename. Historical issue....
        features['gain'] = features['gain_per_action']
        del features['gain_per_action']

        features['action'] = tf.FixedLenFeature([], tf.string)
        features['cur_prob'] = tf.FixedLenFeature([], tf.float32)
        features['next_prob'] = tf.FixedLenFeature([], tf.float32)
        return features

    def _handle_specific_feature(self, result, parsed_features):
        # result['actions'] = tf.cast(
        #     tf.decode_raw(parsed_features['action'], tf.uint8), tf.int32)
        result['actions'] = tf.decode_raw(parsed_features['action'], tf.uint8)
        result['gain'] = tf.decode_raw(parsed_features['gain'], tf.float32)[0]
        result['cur_prob'] = parsed_features['cur_prob']
        result['next_prob'] = parsed_features['next_prob']


class MIMIC_cache_discretized_joint_exp_independent_measurement(MIMIC_cache_discretized_exp):
    '''
    Deal with experience generated by class MIMIC_discretized_joint_exp_independent_measurement
    '''

    def _parse_function(self, example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),

            'cur_action': tf.FixedLenFeature([], tf.string),
            'next_action': tf.FixedLenFeature([], tf.string),

            'gain_per_action': tf.FixedLenFeature([], tf.string),
            'prob_gain_per_action': tf.FixedLenFeature([], tf.string),
            'std_gain_per_action': tf.FixedLenFeature([], tf.string),

            'gain_joint': tf.FixedLenFeature([], tf.float32),
            'prob_joint': tf.FixedLenFeature([], tf.float32),
            'std_joint': tf.FixedLenFeature([], tf.float32),
            'prob_null': tf.FixedLenFeature([], tf.float32),
            'std_null': tf.FixedLenFeature([], tf.float32),
            'prev_prob_joint': tf.FixedLenFeature([], tf.float32),
            'prev_std_joint': tf.FixedLenFeature([], tf.float32),

            'labels': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
            'mortality': tf.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        result = {}
        for k in ['cur_state', 'next_state', 'gain_per_action',
                  'prob_gain_per_action', 'std_gain_per_action']:
            result[k] = tf.decode_raw(parsed_features[k], tf.float32)

        # Convert the image data from string back to the numbers
        result['cur_action'] = tf.cast(tf.decode_raw(parsed_features['cur_action'], tf.uint8), tf.float32)
        result['next_action'] = tf.cast(tf.decode_raw(parsed_features['next_action'], tf.uint8), tf.float32)

        # Add others without decoding
        for k in ['labels', 'patient_inds', 'the_steps', 'total_steps',  'gain_joint', 'mortality',
                  'prob_joint', 'std_joint', 'prob_null', 'std_null', 'prev_prob_joint', 'prev_std_joint']:
            result[k] = parsed_features[k]

        return result


class MIMIC_cache_discretized_joint_exp_random_order(MIMIC_cache_discretized_exp):
    '''
    Deal with experience generated by class MIMIC_discretized_joint_exp_independent_measurement
    '''

    def _parse_function(self, example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'delay_update_state': tf.FixedLenFeature([], tf.string),
            'cur_history': tf.FixedLenFeature([], tf.string),
            'next_history': tf.FixedLenFeature([], tf.string),
            'actions': tf.FixedLenFeature([], tf.string),

            'prob_gain': tf.FixedLenFeature([], tf.float32),
            'cur_prob': tf.FixedLenFeature([], tf.float32),
            'next_prob': tf.FixedLenFeature([], tf.float32),

            'labels': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
            'mortality': tf.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        result = {}
        for k in ['cur_state', 'next_state', 'delay_update_state']:
            result[k] = tf.decode_raw(parsed_features[k], tf.float32)

        for k in ['cur_history', 'next_history', 'actions']:
            result[k] = tf.cast(tf.decode_raw(parsed_features[k], tf.uint8), tf.float32)

        # Add others without decoding
        for k in ['labels', 'patient_inds', 'the_steps', 'total_steps', 'prob_gain', 'mortality',
                  'cur_prob', 'next_prob']:
            result[k] = parsed_features[k]

        return result


class MIMIC_cache_discretized_joint_exp_random_order_with_obs(MIMIC_cache_discretized_joint_exp_random_order):
    def _parse_function(self, example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'delay_update_state': tf.FixedLenFeature([], tf.string),
            'cur_history': tf.FixedLenFeature([], tf.string),
            'next_history': tf.FixedLenFeature([], tf.string),
            'actions': tf.FixedLenFeature([], tf.string),

            'prob_gain': tf.FixedLenFeature([], tf.float32),
            'cur_prob': tf.FixedLenFeature([], tf.float32),
            'next_prob': tf.FixedLenFeature([], tf.float32),

            'labels': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
            'mortality': tf.FixedLenFeature([], tf.int64),

            't_1_obs': tf.FixedLenFeature([], tf.string),
            't_2_obs': tf.FixedLenFeature([], tf.string),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        result = {}
        for k in ['cur_state', 'next_state', 'delay_update_state', 't_1_obs', 't_2_obs']:
            result[k] = tf.decode_raw(parsed_features[k], tf.float32)

        for k in ['cur_history', 'next_history', 'actions']:
            result[k] = tf.cast(tf.decode_raw(parsed_features[k], tf.uint8), tf.float32)

        # Add others without decoding
        for k in ['labels', 'patient_inds', 'the_steps', 'total_steps', 'prob_gain', 'mortality',
                  'cur_prob', 'next_prob']:
            result[k] = parsed_features[k]

        return result


class MIMIC_cache_discretized_exp_env_v3(MIMIC_cache_discretized_exp):
    ''' Corresponding to MIMIC_per_time_env_exp '''
    def _parse_function(self, example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'prob_gain': tf.FixedLenFeature([], tf.float32),
            'cur_prob': tf.FixedLenFeature([], tf.float32),
            'next_prob': tf.FixedLenFeature([], tf.float32),
            'cur_actions': tf.FixedLenFeature([], tf.string),
            'next_actions': tf.FixedLenFeature([], tf.string),
            'cur_obs': tf.FixedLenFeature([], tf.string),
            'next_obs': tf.FixedLenFeature([], tf.string),

            'labels': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
            'mortality': tf.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        result = {}
        for k in ['cur_state', 'next_state', 'cur_obs', 'next_obs']:
            result[k] = tf.decode_raw(parsed_features[k], tf.float32)

        for k in ['cur_actions', 'next_actions']:
            result[k] = tf.cast(tf.decode_raw(parsed_features[k], tf.uint8), tf.float32)

        # Add others without decoding
        for k in ['prob_gain', 'cur_prob', 'next_prob', 'labels', 'patient_inds',
                  'the_steps', 'total_steps', 'mortality']:
            result[k] = parsed_features[k]

        return result


class MIMIC_cache_discretized_exp_env_v3_last_obs(MIMIC_cache_discretized_exp):
    ''' Corresponding to MIMIC_per_time_env_exp_last_obs '''
    def _parse_function(self, example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'prob_gain': tf.FixedLenFeature([], tf.float32),
            'cur_prob': tf.FixedLenFeature([], tf.float32),
            'next_prob': tf.FixedLenFeature([], tf.float32),
            'cur_actions': tf.FixedLenFeature([], tf.string),
            'next_actions': tf.FixedLenFeature([], tf.string),
            'cur_obs': tf.FixedLenFeature([], tf.string),
            'next_obs': tf.FixedLenFeature([], tf.string),
            't_1_obs': tf.FixedLenFeature([], tf.string),
            't_2_obs': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
            'mortality': tf.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        result = {}
        for k in ['cur_state', 'next_state', 'cur_obs', 'next_obs', 't_1_obs', 't_2_obs']:
            result[k] = tf.decode_raw(parsed_features[k], tf.float32)

        for k in ['cur_actions', 'next_actions']:
            result[k] = tf.cast(tf.decode_raw(parsed_features[k], tf.uint8), tf.float32)

        # Add others without decoding
        for k in ['prob_gain', 'cur_prob', 'next_prob', 'labels', 'patient_inds',
                  'the_steps', 'total_steps', 'mortality']:
            result[k] = parsed_features[k]

        return result




class MIMIC_cache_discretized_exp_v2(MIMIC_cache_discretized_exp):
    ''' Add in estiamted next obs '''
    @staticmethod
    def _parse_function(example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'cur_prob': tf.FixedLenFeature([], tf.string),
            'next_prob': tf.FixedLenFeature([], tf.string),
            'cur_action': tf.FixedLenFeature([], tf.string),
            'next_action': tf.FixedLenFeature([], tf.string),
            'est_next_obs': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'patient_inds': tf.FixedLenFeature([], tf.int64),
            'the_steps': tf.FixedLenFeature([], tf.int64),
            'total_steps': tf.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        # Convert the image data from string back to the numbers
        cur_state = tf.decode_raw(parsed_features['cur_state'], tf.float32)
        next_state = tf.decode_raw(parsed_features['next_state'], tf.float32)
        cur_prob = tf.decode_raw(parsed_features['cur_prob'], tf.float32)
        next_prob = tf.decode_raw(parsed_features['next_prob'], tf.float32)

        # cur_action = tf.decode_raw(parsed_features['cur_action'], tf.float32)
        # next_action = tf.decode_raw(parsed_features['next_action'], tf.float32)

        cur_action = tf.cast(tf.decode_raw(parsed_features['cur_action'], tf.uint8), tf.float32)
        next_action = tf.cast(tf.decode_raw(parsed_features['next_action'], tf.uint8), tf.float32)

        est_next_obs = tf.decode_raw(parsed_features['est_next_obs'], tf.float32)

        return {'cur_state': cur_state,
                'next_state': next_state,
                'cur_probs': cur_prob,
                'next_probs': next_prob,
                'cur_actions': cur_action,
                'next_actions': next_action,
                'est_next_obs': est_next_obs,
                'labels': parsed_features['label'],
                'patient_ind': parsed_features['patient_inds'],
                'the_step': parsed_features['the_steps'],
                'total_steps': parsed_features['total_steps']}


class MIMIC_cache_discretized_exp_env(MIMIC_cache_discretized_exp):
    ''' Add in estiamted next obs '''
    @staticmethod
    def _parse_function(example_proto):
        features = {
            'cur_state': tf.FixedLenFeature([], tf.string),
            'next_state': tf.FixedLenFeature([], tf.string),
            'cur_action': tf.FixedLenFeature([], tf.string),
            'next_action': tf.FixedLenFeature([], tf.string),
            'cur_obs': tf.FixedLenFeature([], tf.string),
            'next_obs': tf.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        # Convert the image data from string back to the numbers
        cur_state = tf.decode_raw(parsed_features['cur_state'], tf.float32)
        next_state = tf.decode_raw(parsed_features['next_state'], tf.float32)

        cur_action = tf.cast(tf.decode_raw(parsed_features['cur_action'], tf.uint8), tf.float32)
        next_action = tf.cast(tf.decode_raw(parsed_features['next_debugaction'], tf.uint8), tf.float32)

        cur_obs = tf.decode_raw(parsed_features['cur_obs'], tf.float32)
        next_obs = tf.decode_raw(parsed_features['next_obs'], tf.float32)

        return {
            'cur_state': cur_state,
            'next_state': next_state,
            'cur_actions': cur_action,
            'next_actions': next_action,
            'cur_obs': cur_obs,
            'next_obs': next_obs
        }


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # cashe_dir = '../RL_exp_cache/1012-15mins-48hrs-joint-indep-measurement/'
    # cashe_dir = '../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/'
    # cashe_dir = '../RL_exp_cache/1020-15mins-24hrs-joint-random-order/'

    # cashe_dir = '../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/'
    # cashe_dir = '../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn-new/'
    # cashe_dir = '../RL_exp_cache/1023-RNN-debug/'
    # cache_dir = '../RL_exp_cache/1102-15mins-24hrs-joint-indep-measurement-mgp-rnn-all-pat/'
    # cache_dir = '../RL_exp_cache/1103-15mins-24hrs-random-order-mgp-rnn-all-pat'
    # cache_dir = '../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/'
    # cache_dir = '../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled//'
    # cache_dir = '../RL_exp_cache/1124-30mns-24hrs-random-order-rnn-neg_sampled-test'
    # cache_dir = '../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/'
    cache_dir = '../RL_exp_cache/0312-30mins-24hrs-20order-rnn-neg_sampled-with-obs/'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # the_dict['prob_gain_per_action'].sum(axis=0) / the_dict['cur_action'].sum(axis=0)
    # the_dict['cur_action'].mean(axis=0)

    with tf.Session(config=config) as sess:
        # mimic_exp = MIMIC_cache_discretized_exp(cache_dir=cashe_dir, action_dim=40)
        # mimic_exp = MIMIC_cache_discretized_joint_exp_independent_measurement(cache_dir=cache_dir)
        # mimic_exp = MIMIC_cache_discretized_joint_exp_random_order(cache_dir=cache_dir)
        # mimic_exp = MIMIC_cache_discretized_exp_env_v3(cache_dir=cache_dir)
        # mimic_exp = MIMIC_cache_discretized_joint_exp_random_order_with_obs(cache_dir=cache_dir)
        mimic_exp = MIMIC_cache_discretized_exp_env_v3_last_obs(cache_dir=cache_dir)

        #generate_per_action_cost(cashe_dir, mimic_exp)

        # val_loader = mimic_exp.gen_val_experience(sess, batch_size=200, shuffle=False)
        # val_loader = mimic_exp.gen_testing_experience(sess, batch_size=200, shuffle=False)
        # val_loader = mimic_exp.gen_experience('val_per_time', sess, batch_size=200, shuffle=False)
        # val_loader = mimic_exp.gen_experience('val_per_time_env', sess, batch_size=200, shuffle=False)
        val_loader = mimic_exp.gen_experience('testing_per_time_env', sess, batch_size=200, shuffle=False)

        for idx, the_dict in enumerate(val_loader):
            print(idx)
