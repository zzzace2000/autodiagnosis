import copy
import time

import numpy as np
import tensorflow as tf

from arch.MGP_RNN import LossRecorderRegression
from arch.ManyToOneMGP_RNN import ManyToOneMGP_RNN


class ManyToOneRNN(ManyToOneMGP_RNN):
    def __init__(self, rnn_imputation='forward_imputation', **kwargs):
        self.rnn_imputation = rnn_imputation
        kwargs['n_mc_smps'] = 1
        super(ManyToOneRNN, self).__init__(**kwargs)

    def _init_placeholder_output(self):
        return tf.placeholder(tf.int32, [None, None],
                              name='O')  # labels. input is NOT as one-hot encoding; convert at each iter

    def create_placeholder(self):
        self.O = self._init_placeholder_output()
        self.num_obs_times = tf.placeholder(tf.int32, [None])  # number of observation times per encounter
        self.num_obs_values = tf.placeholder(tf.int32, [None])  # number of observation values per encounter
        self.num_rnn_grid_times = tf.placeholder(tf.int32, [None])  # length of each grid to be fed into RNN in batch

        self.target_weights = 1

        self.is_training = tf.placeholder_with_default(False, shape=())
        self.rnn_input_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.rnn_output_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.rnn_state_keep_prob = tf.placeholder_with_default(1.0, shape=())

    def create_gp_params(self):
        pass

    def run_rnn(self):
        self.rnn_inputs = tf.placeholder("float", [None, None, self.input_dim],
                                         name='rnn_inputs')
        self.N = tf.shape(self.rnn_inputs)[0]

        self.outputs, self.states = tf.nn.dynamic_rnn(cell=self.stacked_lstm,
                                                      inputs=self.rnn_inputs,
                                                      dtype=tf.float32)

    def pad_measurements(self, Ys, Ts, ind_kts, ind_kfs, Xs, labels=None, covs=None,
                         **kwargs):
        the_func = getattr(self, self.rnn_imputation)

        # Remove covs
        if self.n_covs == 0:
            covs = None

        measurements, labels = the_func(
            self.num_features, self.add_missing,
            Ys, Ts, ind_kts, ind_kfs, Xs, labels, covs=covs)

        return {'rnn_inputs': measurements, 'labels': labels}

    def create_feed_dict(self, rnn_inputs, labels, **kwargs):
        feed_dict = {self.rnn_inputs: rnn_inputs, self.O: labels}
        return feed_dict

    def _monitor_params(self):
        ''' Remove monitoring GP length '''
        return {}

    @classmethod
    def forward_imputation(cls, num_features, add_missing,
                           Ys, Ts, ind_kts, ind_kfs, Xs, labels,
                           covs=None, **kwargs):
        def forward_impute(measurements, count_measurements, prev_time_forward_imp_val,
                   backward_imp_val):
            # Forward and backward Imputation to get first value
            if (count_measurements[:, 0, :] == 0).sum() > 0:
                if (prev_time_forward_imp_val == -1).sum() > 0:
                    prev_time_forward_imp_val[prev_time_forward_imp_val == -1] = \
                        backward_imp_val[prev_time_forward_imp_val == -1]
                measurements[:, 0, :][count_measurements[:, 0, :] == 0] = \
                    prev_time_forward_imp_val[count_measurements[:, 0, :] == 0]

            # Start doing forward imputation. If met -1, then do a backward imputation!?
            for time_idx in range(1, measurements.shape[1]):
                arr = count_measurements[:, time_idx, :]
                if (arr == 0).sum() > 0:
                    measurements[:, time_idx, :][arr == 0] = \
                        measurements[:, (time_idx - 1), :][arr == 0]
            return measurements

        return cls._impute(num_features, add_missing,
                           Ys, Ts, ind_kts, ind_kfs, Xs, labels, forward_impute,
                           covs)

    @classmethod
    def mean_imputation(cls, num_features, add_missing,
                        Ys, Ts, ind_kts, ind_kfs, Xs, labels,
                        covs=None, **kwargs):
        def mean_impute(measurements, count_measurements,
                        prev_time_forward_imp_val, backward_imp_val):
            return measurements

        return cls._impute(num_features, add_missing,
                           Ys, Ts, ind_kts, ind_kfs, Xs, labels, mean_impute,
                           covs)

    @classmethod
    def _impute(cls, num_features, add_missing,
                Ys, Ts, ind_kts, ind_kfs, Xs, labels, impute_func,
                covs=None):
        ''' Discretize based on X intervals. Will get (x_max - 1) measurements! '''

        assert len(Ys) == len(Ts) == len(ind_kts) == len(ind_kfs) == len(Xs), \
            str((len(Ys), len(Ts), len(ind_kts), len(ind_kfs), len(Xs)))

        # Assume all the batch have same X interval
        X_interval = Xs[0][1] - Xs[0][0]

        x_len, x_max_len, x_pad = cls._get_lens_info_from_batch(Xs)

        sum_measurements = np.zeros((len(Ys), x_max_len - 1, num_features))
        count_measurements = np.zeros((len(Ys), x_max_len - 1, num_features))

        prev_time_forward_imp_val = -np.ones((len(Ys), num_features))
        backward_imp_val = np.zeros((len(Ys), num_features))

        for i, (Y, T, ind_kf, ind_kt, X) in enumerate(zip(Ys, Ts, ind_kfs, ind_kts, Xs)):
            # Loop through all the measurements
            for y, ind_f, ind_t in zip(Y, ind_kf, ind_kt):
                if T[ind_t] < (X[0]):
                    prev_time_forward_imp_val[i, ind_f] = y
                    continue

                the_interval = int((T[ind_t] - (X[0])) // X_interval)

                # Handle edge case: if it's in the last time point, give it to last interval
                if the_interval == x_max_len - 1 and (T[ind_t] - (X[0])) == the_interval * X_interval:
                    the_interval -= 1

                sum_measurements[i, the_interval, ind_f] += y
                count_measurements[i, the_interval, ind_f] += 1

                if backward_imp_val[i, ind_f] == 0:
                    backward_imp_val[i, ind_f] = y

        measurements = np.divide(sum_measurements, count_measurements,
                                 out=np.zeros_like(sum_measurements),
                                 where=(count_measurements != 0))

        # Do the imputation here
        measurements = impute_func(measurements, count_measurements,
                                   prev_time_forward_imp_val, backward_imp_val)

        concatenate_arr = [measurements]

        if covs is not None and len(covs) > 0:
            if not isinstance(covs, np.ndarray):  # It needs padding
                covs = np.array(covs)
            the_covs = np.tile(np.expand_dims(covs, 1), (1, x_max_len - 1, 1))
            concatenate_arr.append(the_covs)

        # missingness
        if add_missing:
            missingness = copy.copy(count_measurements)
            missingness[missingness > 0] = 1
            concatenate_arr.append(missingness)

        measurements = np.concatenate(concatenate_arr, axis=-1)
        return measurements, labels
