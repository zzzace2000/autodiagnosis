import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

from utils.general import log


class MGP_RNN:
    def __init__(self, learning_rate=0.001, l2_penalty=1e-2, n_hidden=100, n_layers=5,
                 n_classes=2, num_features=10, n_meds=0, n_covs=0, n_mc_smps=25,
                 add_missing=False, use_target_weight=False, logfile=None,
                 invert_missing=False, **kwargs):

        # Learning Parameters
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

        # Network Parameters
        self.n_hidden = n_hidden  # hidden layer num of features; assumed same
        self.n_layers = n_layers  # number of layers of stacked LSTMs
        self.n_classes = n_classes  # binary outcome
        self.num_features = num_features

        self.n_meds = n_meds
        self.n_covs = n_covs
        self.invert_missing = invert_missing
        self.add_missing = add_missing
        if self.add_missing:
            self.n_meds += num_features

        self.use_med_cov = (self.n_meds + self.n_covs) > 0

        self.input_dim = self.num_features + self.n_meds + self.n_covs  # dimensionality of input sequence.
        self.n_mc_smps = n_mc_smps

        self.use_target_weight = use_target_weight
        self.logfile = logfile

        self.create_placeholder()
        self.create_gp_params()
        self.create_rnn_params()
        self.build_graph()

    @classmethod
    def load_saved_mgp_rnn(cls, sess, model_dir):
        hyper_path = os.path.join(model_dir, 'hyperparams.log')
        if not os.path.exists(hyper_path):
            raise FileNotFoundError('No file found in the path %s' % hyper_path)

        params = json.load(open(hyper_path, 'rb'))

        if 'rnn_cls' in params['classifier']:
            assert params['classifier']['rnn_cls'] == cls.__name__

        # Set up classifier ------------------------------------------
        with tf.variable_scope('mgp_rnn', reuse=tf.AUTO_REUSE):
            mgp_rnn = cls(**params['classifier'])

        # Best model idx
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mgp_rnn'))
        if os.path.exists(os.path.join(model_dir, 'model.log')):
            best_model_idx = json.load(open(os.path.join(model_dir, 'model.log')))['best_model_idx']
            print('Find model.log ... The best model idx is %d' % best_model_idx)
            saver.restore(sess, os.path.join(model_dir, 'model-%d' % best_model_idx))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        return mgp_rnn

    @staticmethod
    def SE_kernel(length, x1, x2):
        x1 = tf.reshape(x1, [-1, 1])  # colvec
        x2 = tf.reshape(x2, [1, -1])  # rowvec
        K = tf.exp(-tf.pow(x1 - x2, 2.0) / length)
        return K

    @staticmethod
    def OU_kernel(length, x1, x2):
        x1 = tf.reshape(x1, [-1, 1])  # colvec
        x2 = tf.reshape(x2, [1, -1])  # rowvec
        K = tf.exp(-tf.abs(x1 - x2) / length)
        return K

    def _init_placeholder_output(self):
        return tf.placeholder(tf.int32, [None, None],
                                name='O')  # labels. input is NOT as one-hot encoding; convert at each iter

    def create_placeholder(self):
        # observed values, times, inducing times
        #  padded to longest in the batch
        self.Y = tf.placeholder("float", [None, None], name='Y')  # batchsize x batch_maxdata_length
        self.T = tf.placeholder("float", [None, None], name='T')  # batchsize x batch_maxdata_length
        self.X = tf.placeholder("float", [None, None], name='X')  # grid points. batchsize x batch_maxgridlen

        self.ind_kf = tf.placeholder(tf.int32, [None, None], name='ind_kf')  # index tasks in Y vector
        self.ind_kt = tf.placeholder(tf.int32, [None, None], name='ind_kt')  # index inputs in Y vector

        if self.use_med_cov:
            self.med_cov_grid = tf.placeholder("float", [None, None, self.n_meds + self.n_covs])  # combine w GP smps to feed into RNN

        self.O = self._init_placeholder_output()
        self.num_obs_times = tf.placeholder(tf.int32, [None])  # number of observation times per encounter
        self.num_obs_values = tf.placeholder(tf.int32, [None])  # number of observation values per encounter
        self.num_rnn_grid_times = tf.placeholder(tf.int32, [None])  # length of each grid to be fed into RNN in batch

        if self.use_target_weight:
            self.target_weights = tf.placeholder("float", [None, None], name='target_weights')
        else:
            self.target_weights = 1.

        self.is_training = tf.placeholder_with_default(False, shape=())
        self.rnn_input_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.rnn_output_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.rnn_state_keep_prob = tf.placeholder_with_default(1.0, shape=())

    def create_gp_params(self):
        # in fully separable case all labs share same time-covariance. Start with 4
        log_length = tf.Variable(tf.random_normal([1], mean=1.35, stddev=0.1),
                                  name="GP-log-length")
        self.length = tf.exp(log_length)

        # different noise level of each lab
        log_noises = tf.Variable(tf.random_normal([self.num_features], mean=-2, stddev=0.1),
                                 name="GP-log-noises")
        self.noises = tf.exp(log_noises)

        # init cov between labs
        L_f_init = tf.Variable(tf.eye(self.num_features),
                               name="GP-Lf")
        Lf = tf.matrix_band_part(L_f_init, -1, 0)
        self.Kf = tf.matmul(Lf, tf.transpose(Lf))

    def create_rnn_params(self):
        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.n_hidden),
                                           input_keep_prob=self.rnn_input_keep_prob,
                                           output_keep_prob=self.rnn_output_keep_prob,
                                           state_keep_prob=self.rnn_state_keep_prob) for _ in range(self.n_layers)])

        # Weights at the last layer given deep LSTM output
        self.out_weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes],
                                                        stddev=0.1),
                                       name="Softmax/W")
        self.out_biases = tf.Variable(tf.random_normal([self.n_classes],
                                                       stddev=0.1),
                                      name="Softmax/b")

    def build_graph(self):
        if self.use_target_weight:
            self.target_weights_dupe = tf.reshape(tf.tile(self.target_weights, [1, self.n_mc_smps]),
                                                  [self.N * self.n_mc_smps * self.max_x_len])
        else:
            self.target_weights_dupe = 1

        # Calculate predictions and stat -----------------
        self.run_rnn()
        self.calculate_predictions_and_probability()

        # Define loss -----------------
        self.fit_loss()

        # Perform optimization -----------------
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)

        self.lr_node = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
                                                  decay_steps=1, decay_rate=0.1)
        optimizer = tf.train.AdamOptimizer(self.lr_node)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def decrease_lr(self, sess):
        orig_lr = sess.run(self.lr_node)
        sess.run(self.add_global)
        next_lr = sess.run(self.lr_node)

        print('Decrease lr from %.1e to %.1e' % (orig_lr, next_lr))

    def draw_GP(self, Yi, Ti, Xi, ind_kfi, ind_kti):
        """
        given GP hyperparams and data values at observation times, draw from
        conditional GP

        inputs:
            length, noises, Lf, Kf: GP params
            Yi: observation values
            Ti: observation times
            Xi: grid points (new times for rnn)
            ind_kfi, ind_kti: indices into Y

        returns:
            draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
        """
        # Cache some calculations
        grid_f = tf.meshgrid(ind_kfi, ind_kfi)  # same as np.meshgrid

        K_tt = self.OU_kernel(self.length, Ti, Ti)
        D = tf.diag(self.noises)

        Kf_big = tf.gather_nd(self.Kf, tf.stack((grid_f[0], grid_f[1]), -1))

        grid_t = tf.meshgrid(ind_kti, ind_kti)
        Kt_big = tf.gather_nd(K_tt, tf.stack((grid_t[0], grid_t[1]), -1))

        Kf_Ktt = tf.multiply(Kf_big, Kt_big)

        DI_big = tf.gather_nd(D, tf.stack((grid_f[0], grid_f[1]), -1))
        DI = tf.diag(tf.diag_part(DI_big))  # D kron I

        # data covariance.
        ny = tf.shape(Yi)[0]
        Ky = Kf_Ktt + DI + 1e-6 * tf.eye(ny)

        ### build out cross-covariances and covariance at grid
        nx = tf.shape(Xi)[0]

        K_xx = self.OU_kernel(self.length, Xi, Xi)
        K_xt = self.OU_kernel(self.length, Xi, Ti)

        ind = tf.concat([tf.tile([i], [nx]) for i in range(self.num_features)], 0)
        grid = tf.meshgrid(ind, ind)
        Kf_big = tf.gather_nd(self.Kf, tf.stack((grid[0], grid[1]), -1))
        ind2 = tf.tile(tf.range(nx), [self.num_features])
        grid2 = tf.meshgrid(ind2, ind2)
        Kxx_big = tf.gather_nd(K_xx, tf.stack((grid2[0], grid2[1]), -1))

        K_ff = tf.multiply(Kf_big, Kxx_big)  # cov at grid points

        full_f = tf.concat([tf.tile([i], [nx]) for i in range(self.num_features)], 0)
        grid_1 = tf.meshgrid(full_f, ind_kfi, indexing='ij')
        Kf_big = tf.gather_nd(self.Kf, tf.stack((grid_1[0], grid_1[1]), -1))

        full_x = tf.tile(tf.range(nx), [self.num_features])
        grid_2 = tf.meshgrid(full_x, ind_kti, indexing='ij')
        Kxt_big = tf.gather_nd(K_xt, tf.stack((grid_2[0], grid_2[1]), -1))

        K_fy = tf.multiply(Kf_big, Kxt_big)

        y_ = tf.reshape(Yi, [-1, 1])
        Ly = tf.cholesky(Ky)
        Mu = tf.matmul(K_fy, tf.cholesky_solve(Ly, y_))

        xi = tf.random_normal((nx * self.num_features, self.n_mc_smps))
        Sigma = K_ff - tf.matmul(K_fy, tf.cholesky_solve(Ly, tf.transpose(K_fy))) + \
                1e-6 * tf.eye(tf.shape(K_ff)[0])
        draw = Mu + tf.matmul(tf.cholesky(Sigma), xi)
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw), [self.n_mc_smps, self.num_features, nx]),
                                    perm=[0, 2, 1])

        return draw_reshape

    def get_GP_samples(self):
        """
        Returns samples from GP at evenly-spaced gridpoints given paded datapoints
        """

        Z = tf.zeros([0, self.max_x_len, self.input_dim])

        # setup tf while loop (have to use this bc loop size is variable)
        def cond(i, Z):
            return i < self.N

        def body(i, Z):
            Yi = tf.reshape(tf.slice(self.Y, [i, 0], [1, self.num_obs_values[i]]), [-1])
            Ti = tf.reshape(tf.slice(self.T, [i, 0], [1, self.num_obs_times[i]]), [-1])
            ind_kfi = tf.reshape(tf.slice(self.ind_kf, [i, 0], [1, self.num_obs_values[i]]), [-1])
            ind_kti = tf.reshape(tf.slice(self.ind_kt, [i, 0], [1, self.num_obs_values[i]]), [-1])
            Xi = tf.reshape(tf.slice(self.X, [i, 0], [1, self.num_rnn_grid_times[i]]), [-1])
            X_len = self.num_rnn_grid_times[i]
            #T_len = self.num_obs_times[i]

            GP_draws = self.draw_GP(Yi, Ti, Xi, ind_kfi, ind_kti)
            pad_len = self.max_x_len - X_len  # pad by this much
            padded_GP_draws = tf.concat([GP_draws, tf.zeros((self.n_mc_smps, pad_len, self.num_features))], 1)

            if self.use_med_cov:
                medcovs = tf.slice(self.med_cov_grid, [i, 0, 0], [1, -1, -1])
                tiled_medcovs = tf.tile(medcovs, [self.n_mc_smps, 1, 1])
                padded_GP_draws = tf.concat([padded_GP_draws, tiled_medcovs], 2)

            Z = tf.concat([Z, padded_GP_draws], 0)

            return i + 1, Z

        i = tf.constant(0)
        i, Z = tf.while_loop(cond, body, loop_vars=[i, Z],
                             shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None])])

        Z.set_shape([None, None, self.input_dim])  # somehow lost shape info, but need this

        return Z

    def run_rnn(self):
        self.N = tf.shape(self.Y)[0]
        self.max_x_len = tf.shape(self.X)[1]

        self.rnn_inputs = self.get_GP_samples()  # batchsize * (num_MC x batch_maxseqlen) * num_inputs
        self.Z = tf.reshape(self.rnn_inputs, [self.N, self.n_mc_smps, self.max_x_len, self.input_dim])

        # duplicate each entry of seqlens, to account for multiple MC samples per observation
        seqlen_dupe = tf.reshape(tf.tile(tf.expand_dims(self.num_rnn_grid_times, 1),
                                         [1, self.n_mc_smps]),
                                 [self.N * self.n_mc_smps])

        self.outputs, self.states = tf.nn.dynamic_rnn(cell=self.stacked_lstm,
                                                      inputs=self.rnn_inputs,
                                                      dtype=tf.float32,
                                                      sequence_length=seqlen_dupe)

    def _get_loss_reg(self):
        loss_reg = self.l2_penalty * tf.reduce_sum(tf.square(self.out_weights))
        for i in range(self.n_layers):
            loss_reg += self.l2_penalty * tf.reduce_sum(
                tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_' + str(i) +
                                          '/basic_lstm_cell/kernel')))

        return loss_reg
    def _get_loss_fit(self):
        O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(self.O, [1, self.n_mc_smps]),
                                              [self.N * self.n_mc_smps * self.max_x_len]), self.n_classes)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.raw_preds,
                                                               labels=O_dupe_onehot) * \
                       self.target_weights_dupe) / tf.cast(self.N, tf.float32)

    def fit_loss(self):
        self.loss_fit = self._get_loss_fit()

        self.loss_reg = self._get_loss_reg()

        self.loss = self.loss_fit + self.loss_reg

    def calculate_predictions_and_probability(self):
        self.raw_preds = tf.matmul(tf.reshape(self.outputs, [self.N * self.n_mc_smps * self.max_x_len, self.n_hidden]),
                                   self.out_weights) + self.out_biases
        self.probs = tf.reshape(tf.nn.softmax(self.raw_preds, -1),
                                [self.N, self.n_mc_smps, self.max_x_len, self.n_classes])
        self.avg_probs = tf.reduce_mean(self.probs, axis=1)
        self.O_pred = tf.argmax(self.avg_probs, -1, output_type=tf.int32)

    def create_feed_dict_by_sparse_arr(self, **kwargs):
        data_dict = self.pad_measurements(**kwargs)
        return self.create_feed_dict(**data_dict)

    def pad_measurements(self, Ys, Ts, ind_kts, ind_kfs, Xs, labels=None, covs=None,
                         **kwargs):
        assert len(Ys) == len(Ts) == len(ind_kts) == len(ind_kfs) == len(Xs), \
            str((len(Ys), len(Ts), len(ind_kts), len(ind_kfs), len(Xs)))

        y_len, y_max_len, y_pad = self._get_lens_info_from_batch(Ys)
        t_len, t_max_len, t_pad = self._get_lens_info_from_batch(Ts)
        _, _, ind_kt_pad = self._get_lens_info_from_batch(ind_kts)
        _, _, ind_kf_pad = self._get_lens_info_from_batch(ind_kfs)
        x_len, x_max_len, x_pad = self._get_lens_info_from_batch(Xs)

        result_dict = {
            'T_pad'     : t_pad, 'Y_pad': y_pad, 'ind_kf_pad': ind_kf_pad,
            'ind_kt_pad': ind_kt_pad, 'X_pad': x_pad, 'T_len': t_len,
            'Y_len'     : y_len, 'X_len': x_len,
            # 'meds_cov_pad ': meds_cov_pad,
            # 'target_weights': target_weights,
            # 'target_weights_pad': target_weights_pad,
        }

        if labels is not None:
            if not isinstance(labels, np.ndarray):  # It needs padding
                _, _, labels = self._get_lens_info_from_batch(labels)
        result_dict['labels'] = labels

        med_cov_pad = None
        if covs is not None and len(covs) > 0 and self.n_covs > 0:
            if not isinstance(covs, np.ndarray):  # It needs padding
                covs = np.array(covs)

            assert covs.shape[-1] == self.n_covs, \
                'Pad shape %d is different from the specified covs %d' % \
                (covs.shape[-1], self.n_covs)

            med_cov_pad = np.tile(np.expand_dims(covs, 1), (1, x_max_len, 1))

        if self.add_missing:
            missing_pad = self._get_missing(Ts, ind_kts, ind_kfs, Xs)

            med_cov_pad = missing_pad if med_cov_pad is None \
                else np.concatenate([med_cov_pad, missing_pad], axis=-1)

        result_dict['meds_cov_pad'] = med_cov_pad
        return result_dict

    @staticmethod
    def _get_lens_info_from_batch(measurements):
        measurements_len = np.array([len(yi) for yi in measurements])
        measurements_maxlen = np.max([np.max(measurements_len), 1])

        num_inds = len(measurements)

        y_pad = np.zeros((num_inds, measurements_maxlen))
        for i in range(num_inds):
            y_pad[i, :measurements_len[i]] = measurements[i]

        return measurements_len, measurements_maxlen, y_pad

    def create_feed_dict(self, T_pad, Y_pad, ind_kf_pad, ind_kt_pad,
                         X_pad, T_len, Y_len, X_len, labels=None,
                         target_weights=None, meds_cov_pad=None, **kwargs):
        feed_dict = {self.Y: Y_pad, self.T: T_pad, self.ind_kf: ind_kf_pad, self.ind_kt: ind_kt_pad,
                     self.X: X_pad, self.num_obs_times: T_len,
                     self.num_obs_values: Y_len, self.num_rnn_grid_times: X_len}
        if labels is not None:
            feed_dict[self.O] = labels

        if self.use_target_weight and target_weights is not None:
            feed_dict[self.target_weights] = target_weights
        if self.use_med_cov and meds_cov_pad is not None:
            feed_dict[self.med_cov_grid] = meds_cov_pad

        return feed_dict

    def _get_missing(self, Ts, ind_kts, ind_kfs, Xs):
        '''
        Get missing value by seeing if the interval btw [-int / 2, int / 2]
        '''

        # Assume all the batch have same X interval
        X_interval = Xs[0][1] - Xs[0][0]

        x_len, x_max_len, x_pad = self._get_lens_info_from_batch(Xs)

        missingness = np.zeros((len(Xs), x_max_len, self.num_features))

        for i, (T, ind_kf, ind_kt, X) in enumerate(zip(Ts, ind_kfs, ind_kts, Xs)):
            # Loop through all the measurements
            for ind_f, ind_t in zip(ind_kf, ind_kt):
                if T[ind_t] < (X[0] - X_interval / 2) or ind_f >= self.num_features:
                    continue
                the_interval = int((T[ind_t] - (X[0] - X_interval / 2)) // X_interval)
                if the_interval == x_max_len: # Handle edge case when time exactly matches
                    the_interval -= 1
                missingness[i, the_interval, ind_f] = 1

        if self.invert_missing:
            missingness = 1. - missingness

        return missingness

    def eval_train_loader(self, sess, loader, batch_print=None,
                          rnn_input_keep_prob=0.7, rnn_output_keep_prob=0.5,
                          rnn_state_keep_prob=0.5):
        # TODO: Support target weights functionality.
        assert self.use_target_weight is False
        recorder = LossRecorder(is_train=True, batch_print=batch_print, logfile=self.logfile)

        return self._eval_loader(sess, loader, recorder, is_train=True,
                                 rnn_input_keep_prob=rnn_input_keep_prob,
                                 rnn_output_keep_prob=rnn_output_keep_prob,
                                 rnn_state_keep_prob=rnn_state_keep_prob)

    def eval_test_loader(self, sess, loader, batch_print=None):
        recorder = LossRecorder(is_train=False, batch_print=batch_print, logfile=self.logfile)

        return self._eval_loader(sess, loader, recorder, is_train=False)

    def _eval_loader(self, sess, loader, recorder, is_train=True,
                     rnn_input_keep_prob=0.7, rnn_output_keep_prob=0.5, rnn_state_keep_prob=0.5):
        total_batches = next(loader)

        for batch, data_dict in enumerate(loader):
            batch_start_time = time.time()
            feed_dict = self.create_feed_dict_by_sparse_arr(**data_dict)

            metrics = [self.loss_fit, self.loss_reg, self.O_pred, self.avg_probs]
            monitor_params = self._monitor_params()
            metrics += list(monitor_params.values())

            if is_train:
                feed_dict[self.rnn_input_keep_prob] = rnn_input_keep_prob
                feed_dict[self.rnn_output_keep_prob] = rnn_output_keep_prob
                feed_dict[self.rnn_state_keep_prob] = rnn_state_keep_prob
                feed_dict[self.is_training] = True
                metrics.append(self.train_op)

            try:
                result = sess.run(metrics, feed_dict)
            except tf.errors.InvalidArgumentError as e:
                print('Discard this batch due to following error:', e)
                continue

            loss_fit_, loss_reg_, O_pred, avg_prob = result[:4]

            for i, k in enumerate(monitor_params.keys()):
                monitor_params[k] = result[4 + i]

            recorder.record(batch, total_batches, batch_start_time,
                            loss_fit_, loss_reg_, O_pred, avg_prob,
                            data_dict['labels'],
                            **monitor_params)

        return recorder.report()

    def _monitor_params(self):
        return {'gp_length': self.length}


class LossRecorder:
    def __init__(self, is_train=True, batch_print=None, logfile=None):
        self.is_train = is_train
        self.batch_print = batch_print
        self.logfile = logfile
        self._set_up_placeholder()

    def _set_up_placeholder(self):
        self.total_loss, self.total_reg, self.total_correct, self.total_num, \
            self.total_pos, self.total_pos_correct = 0, 0, 0, 0, 0, 0
        self.all_probs, self.all_labels = [], []

    def record(self, batch, total_batches, batch_start_time,
               loss_fit_, loss_reg_, O_pred, avg_prob, labels, **kwargs):
        self.total_loss += loss_fit_
        self.total_reg += loss_reg_
        if isinstance(labels, list):
            labels = np.array(labels, dtype=int)

        self.total_correct += (O_pred == labels).sum()
        self.total_num += len(labels)

        self.total_pos_correct += ((O_pred == labels)[labels == 1]).sum()
        self.total_pos += labels.sum()

        # Record it only if testing
        if not self.is_train:
            self.all_probs.append(avg_prob[:, :, 1].reshape(-1, ))
            self.all_labels.append(labels.reshape(-1, ))

        if self.batch_print is not None and batch % self.batch_print == (self.batch_print - 1):
            positivity = 0 if self.total_pos == 0 else self.total_pos_correct / self.total_pos

            kwargs_print = ', {}'.format(kwargs) if len(kwargs) > 0 else ''
            log(self.logfile,
                "Batch [{:d}/{:d}] loss_fit: {:.1e} loss_reg: {:.1e} "
                "avg_acc: {:.3f} true_positive: {:.3f}"
                " time: {:.2f}s{}".format(
                batch + 1, total_batches, self.total_loss / self.total_num,
                self.total_reg / self.total_num, self.total_correct / self.total_num,
                    positivity, time.time() - batch_start_time, kwargs_print))
            sys.stdout.flush()

    def report(self):
        loss = self.total_loss / self.total_num
        reg_loss = self.total_reg / self.total_num
        acc = self.total_correct / self.total_num
        true_positive = self.total_pos_correct / self.total_pos

        result_dict = dict(loss=loss, reg_loss=reg_loss, acc=acc, true_positive=true_positive,
                           total_num=self.total_num)

        if self.is_train:
            return result_dict

        all_labels = np.concatenate(self.all_labels, axis=0)
        all_probs = np.concatenate(self.all_probs, axis=0)

        result_dict['auroc'] = roc_auc_score(all_labels, all_probs)
        result_dict['aupr'] = average_precision_score(all_labels, all_probs)

        return result_dict


class LossRecorderRegression(LossRecorder):
    def _set_up_placeholder(self):
        self.total_loss, self.total_reg, self.total_num = 0, 0, 0

        self.all_outputs, self.all_targets = [], []

    def record(self, batch, total_batches, batch_start_time,
               loss_fit_, loss_reg_, outputs, targets, **kwargs):

        self.total_loss += loss_fit_
        self.total_reg += loss_reg_
        self.total_num += len(targets)

        if not self.is_train:
            self.all_outputs.append(outputs)
            self.all_targets.append(targets)

        if self.batch_print is not None and batch % self.batch_print == (
                self.batch_print - 1):
            kwargs_print = ', {}'.format(kwargs) if len(kwargs) > 0 else ''

            log(self.logfile,
                "Batch [{:d}/{:d}] loss_fit: {:.1e} loss_reg: {:.1e} "
                " time: {:.2f}s{}".format(
                    batch + 1, total_batches, self.total_loss / self.total_num,
                    self.total_reg / self.total_num,
                    time.time() - batch_start_time, kwargs_print))

            sys.stdout.flush()

    def report(self):
        loss = self.total_loss / self.total_num
        reg_loss = self.total_reg / self.total_num
        result_dict = dict(loss=loss, reg_loss=reg_loss)

        if self.is_train:
            return result_dict

        all_targets = np.concatenate(self.all_targets)
        all_outputs = np.concatenate(self.all_outputs)

        result_dict['r2'] = r2_score(y_true=all_targets, y_pred=all_outputs,
                                     multioutput='variance_weighted')
        return result_dict
