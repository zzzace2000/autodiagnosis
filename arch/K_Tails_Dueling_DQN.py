import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from arch.memory import Memory
from utils.general import build_nn_module


class KTailsDuelingDQN:
    '''
    Doing a K-tailed DQN without any independent NN layer for each action.
    '''

    def __init__(self, variable_scope, state_dim, action_dim,
                 lr, keep_prob, reg_constant, num_shared_all_layers,
                 num_shared_dueling_layers, num_hidden_units,
                 replace_target_batch, memory_size, log=None, normalized_file=None,
                 reward_attributes=None):
        self.variable_scope = variable_scope
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = lr
        self.keep_prob = keep_prob
        self.reg_constant = reg_constant

        self.num_shared_all_layers = num_shared_all_layers
        self.num_shared_dueling_layers = num_shared_dueling_layers
        self.num_hidden_units = num_hidden_units

        self.replace_target_batch = replace_target_batch
        self.memory_size = memory_size
        self.memory = Memory(capacity=memory_size)  # Experience replay
        self.memory_counter = 0
        self.learning_step_counter = 0

        self.log = log if log is not None else print

        self.normalized_file = normalized_file
        if self.normalized_file is not None:
            state_mean_and_std = pd.read_csv(self.normalized_file)
            mean, std = state_mean_and_std['mean'].values, state_mean_and_std['std'].values

            assert len(mean) <= self.state_dim
            self.log('Normalize the state for the first %d dim with total %d dim' % (len(mean), self.state_dim))

            if len(mean) < self.state_dim:
                mean = np.concatenate((mean, np.zeros(self.state_dim - len(mean))), axis=0)
                std = np.concatenate((std, np.ones(self.state_dim - len(std))), axis=0)

            self.state_mean = tf.constant(mean[None, :], dtype=tf.float32)
            self.state_std = tf.constant(std[None, :], dtype=tf.float32)

        self.reward_attributes = reward_attributes

        self._build_model()

    @staticmethod
    def load_saved_model(sess, the_dir, best_model_idx=None):
        with open(os.path.join(the_dir, 'hyperparams.log')) as op:
            hyperparam_dict = json.load(op)

        if best_model_idx is None:
            with open(os.path.join(the_dir, 'results.json'), 'r') as file:
                training_results = json.load(file)
            best_model_idx = training_results['best_model_idx']

        # set up trained policy ----------------
        # compatibility for old version
        if 'reward_attributes' not in hyperparam_dict['rl']:
            hyperparam_dict['rl']['reward_attributes'] = hyperparam_dict['reward']

        from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN, LastObsSequencialDuelingDQN

        # compatibility for old version
        dqn_cls = 'SequencialDuelingDQN' if 'dqn_cls' not in hyperparam_dict else hyperparam_dict['dqn_cls']

        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            dqn_cls = eval(dqn_cls)
            model = dqn_cls(**hyperparam_dict['rl'])

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dqn'))
        saver.restore(sess, os.path.join(the_dir, 'model-%d' % best_model_idx))

        return model

    def _init_placeholders(self):
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.is_weights = tf.placeholder(tf.float32, [None], name='is_weights')  # weights for prioritized exp replay
        self.gamma = tf.placeholder(tf.float32, shape=(), name='gamma')

        self.s = tf.placeholder(tf.float32, [None, self.state_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.state_dim], name='s_')

        # We change the following 2 variables
        self.a = tf.placeholder(tf.int32, [None, self.action_dim], name='a')
        self.r = tf.placeholder(tf.float32, [None, self.action_dim], name='r')

    def _unwarp_memory(self, batch_memory):
        s = batch_memory[:, :self.state_dim]
        a = batch_memory[:, self.state_dim : self.state_dim + self.action_dim]
        r = batch_memory[:, self.state_dim + self.action_dim : self.state_dim + self.action_dim * 2]
        s_ = batch_memory[:, (self.state_dim + self.action_dim * 2):]

        return s, a, r, s_

    def _build_net(self, obs):
        with tf.variable_scope('shared_all_rep'):
            rep = \
                build_nn_module(input=obs, num_layers=self.num_shared_all_layers,
                                num_hidden_units=self.num_hidden_units,
                                reg_constant=self.reg_constant,
                                keep_prob=self.keep_prob,
                                is_training=self.is_training)

        with tf.variable_scope('value'):
            value = build_nn_module(rep, self.num_shared_dueling_layers, self.num_hidden_units,
                                    self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('advantage'):
            adv = build_nn_module(rep, self.num_shared_dueling_layers, self.num_hidden_units,
                                  self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('q'):
            value = tf.contrib.layers.fully_connected(
                value, self.action_dim, activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))

            adv = tf.contrib.layers.fully_connected(
                adv, self.action_dim * 2, activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))

            adv = tf.reshape(tensor=adv, shape=[-1, self.action_dim, 2])  # batch_size x action_dim x 2
            value = tf.stack((value, value), axis=-1)  # batch_size x action_dim x 2

            # Q(s_t, a_t) = V(s_t) + A(s_t, a_t)
            q = value + (adv - tf.reduce_mean(adv, axis=-1, keepdims=True))  # batch_size x action_dim x 2

        return q

    def _get_q_eval(self):
        """ Return the q value corresponding to the whether action being perform or not for each tail """
        batch_size = tf.shape(self.s)[0]
        batch_idx = tf.range(batch_size * self.action_dim)
        q_cur = tf.reshape(self.q_cur, (batch_size * self.action_dim, 2))
        a = tf.reshape(self.a, [batch_size * self.action_dim])
        q_eval = tf.reshape(tf.gather_nd(q_cur, tf.stack((batch_idx, a), axis=-1)), (batch_size, self.action_dim))

        return q_eval  # batch_size x action_dim

    def _get_q_target(self):
        return self.r + self.gamma * tf.reduce_max(self.q_next, axis=-1)  # batch_size x action_dim

    def _get_replace_target_op(self):
        """ Update target net parameters by replacing them evaluation net parameters """
        t_params = tf.trainable_variables(scope=self.variable_scope + '/target_net')
        e_params = tf.trainable_variables(scope=self.variable_scope + '/eval_net')

        return [tf.assign(ref=t, value=e) for t, e in zip(t_params, e_params)]

    def _get_reg_loss(self, scope):
        reg_loss = self.reg_constant * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
        tf.summary.scalar(name='reg_loss', tensor=reg_loss)
        return reg_loss

    def _get_loss(self):
        """ Return the average l2 error for the batch of expereicne """
        batch_size = tf.cast(tf.shape(self.q_target)[0], tf.float32)
        loss = tf.nn.l2_loss(tf.expand_dims(self.is_weights, 1) * (self.q_target - self.q_eval)) / batch_size
        tf.summary.scalar(name='loss', tensor=loss)
        return loss

    def _get_l2_errors(self):
        """ Return l2 error for each expereince """
        return tf.reduce_mean(tf.square(self.q_target - self.q_eval), axis=-1)

    def _get_train_op(self, scope):
        params = tf.trainable_variables(scope=scope)
        return tf.train.AdamOptimizer(self.lr_node).minimize(self.loss + self.reg_loss, var_list=params)

    def _get_train_op_with_grad_summ(self, scope):
        params = tf.trainable_variables(scope=scope)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_node)
        grads, vars = zip(*optimizer.compute_gradients(self.loss + self.reg_loss, var_list=params))
        train_op = optimizer.apply_gradients(zip(grads, vars))
        [tf.summary.histogram("%s-grad" % v.name.replace(':', '_'), g) for g, v in zip(grads, vars)]

        return train_op

    def _get_learning_rate(self):
        lr_node = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=1,
                                             decay_rate=0.5, staircase=True)
        tf.summary.scalar('lr', lr_node)
        return lr_node

    def _get_best_actions(self):
        return tf.argmax(self.q_cur, axis=-1)

    def _build_model(self):
        self._init_placeholders()

        with tf.variable_scope('eval_net'):
            s = self.s
            if self.normalized_file is not None:
                s = (s - self.state_mean) / self.state_std

            self.q_cur = self._build_net(s)
            self.q_eval = self._get_q_eval()

        with tf.variable_scope('target_net'):
            s_ = self.s_
            if self.normalized_file is not None:
                s_ = (s_ - self.state_mean) / self.state_std
            self.q_next = self._build_net(s_)
            self.q_target = self._get_q_target()
            self.replace_target_op = self._get_replace_target_op()

        with tf.variable_scope('lr_decay'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.add_global = self.global_step.assign_add(1)
            self.lr_node = self._get_learning_rate()

        with tf.variable_scope('loss'):
            self.l2_errors = self._get_l2_errors()
            self.reg_loss = self._get_reg_loss(scope=self.variable_scope + '/eval_net')
            self.loss = self._get_loss()

        with tf.variable_scope('train'):
            self._train_op = self._get_train_op(scope=self.variable_scope + '/eval_net')

        self.best_actions = self._get_best_actions()
        self.summary_ops = tf.summary.merge_all()

    def get_best_actions(self, sess, s):
        return sess.run(self.best_actions, feed_dict={self.s: s, self.is_training: False})

    def decrease_lr(self, sess):
        """ decrease learning rate when call """
        orig_lr = sess.run(self.lr_node)
        sess.run(self.add_global)
        next_lr = sess.run(self.lr_node)

        self.log('Decrease lr from {} to {}'.format(orig_lr, next_lr))

    def store_transition(self, s, a, r, s_):
        """ Store current state, action, reward and next state to memory buffer """
        batch_size = s.shape[0]
        transitions = np.hstack((s, a, r, s_))
        for i in range(batch_size):
            self.memory.store(transitions[i, :])

        self.memory_counter += batch_size

    def fit_loader(self, sess, exp_loader, is_training, batch_print=100,
                   get_best_action=False, writer=None, debug=False):
        sum_loss, sum_reg_loss, sum_num = 0, 0, 0

        result = {}

        for batch_idx, exp in enumerate(exp_loader):
            s, a, r, s_, gamma = self.process_exp_func(exp)
            batch_size = len(s)

            if is_training:
                self.store_transition(s, a, r, s_)
                if self.memory_counter < self.memory_size:
                    continue

                if self.learning_step_counter % self.replace_target_batch == 0:
                    self.log('replace target net params with evaluation net params')
                    sess.run(self.replace_target_op)

                self.learning_step_counter += 1
                tree_idx, batch_memory, is_weights = self.memory.sample(batch_size)
                s, a, r, s_ = self._unwarp_memory(batch_memory)
            else:
                is_weights = np.ones(batch_size)

            feed_dict = {self.s          : s,
                         self.s_         : s_,
                         self.a          : a,
                         self.r          : r,
                         self.is_weights : is_weights,
                         self.is_training: is_training,
                         self.gamma      : gamma}

            if is_training:
                _, l2_errors, loss, reg_loss, summary_str = sess.run(
                    [self._train_op, self.l2_errors, self.loss, self.reg_loss, self.summary_ops], feed_dict=feed_dict)
                self.memory.batch_update(tree_idx, l2_errors)  # update priority for experience replay
            else:
                loss, reg_loss, best_action, summary_str = sess.run(
                    [self.loss, self.reg_loss, self.best_actions, self.summary_ops],
                    feed_dict=feed_dict)

                if get_best_action:
                    if 'best_actions' not in result:
                        result['best_actions'] = []
                    result['best_actions'].append(best_action)

            if writer is not None and is_training:
                writer.add_summary(summary_str, self.learning_step_counter)
                writer.flush()

            sum_loss += loss * batch_size
            sum_reg_loss += reg_loss * batch_size
            sum_num += batch_size

            if batch_print is not None and batch_idx % batch_print == 0:
                self.log('Batch {}: loss {:.3}, reg: {:.3}'.format(
                    batch_idx, sum_loss / sum_num, sum_reg_loss / sum_num))

            if debug and batch_idx > 100:
                break

        # Handle writer in validation that only outputs graph / specific loss
        if writer is not None and is_training is False:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=sum_loss * 1.0 / sum_num),
                tf.Summary.Value(tag="reg_loss", simple_value=sum_reg_loss * 1.0 / sum_num),
            ])
            writer.add_summary(summary, self.learning_step_counter)

            if get_best_action:
                values = np.concatenate(result['best_actions'])

                # summary = self._get_tf_image_summary_with_histogram(sess, values)
                summary = self._get_tf_histogram_summary(values, tag='actions_hist')
                writer.add_summary(summary, self.learning_step_counter)

            writer.flush()

        result.update({'loss': sum_loss * 1.0 / sum_num, 'reg_loss': sum_reg_loss * 1.0 / sum_num,
                       'total_num': sum_num})
        return result

    def _get_tf_histogram_summary(self, values, tag, max_val=None):
        if max_val is None:
            max_val = self.action_dim + 1

        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=np.arange(max_val), normed=True)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        return summary

    def _get_tf_image_summary_with_histogram(self, sess, values, tag='plot', max_val=None):
        import matplotlib.pyplot as plt
        if max_val is None:
            max_val = self.action_dim + 1

        plt.close()
        plt.hist(values, bins=np.arange(max_val), density=True, stacked=True)
        with io.BytesIO() as plot_buf:
            plt.savefig(plot_buf, format='png')
            plot_buf.seek(0)

            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)

            # Add image summary
            summary_op = tf.summary.image(tag, image)
            summary = sess.run(summary_op)

            plt.close()

        return summary

    def _reward_func(self, exp):
        # If mortality is 1, then encourage to increase probability.
        # If mortality is 0, be negative.
        information_gain = self.reward_attributes['gain_coef'] * exp['prob_gain_per_action'] * (2 * exp['labels'][:, None] - 1)

        # action cost.
        action_cost = exp['cur_action']

        rewards = information_gain - self.reward_attributes['action_cost_coef'] * action_cost
        return rewards

    def process_exp_func(self, exp):
        r = self._reward_func(exp)

        s = exp['cur_state']
        s_ = exp['next_state']
        a = exp['cur_action']
        # assert s.shape[1] == args.rl_state_dim, str(s.shape) + str(args.rl_state_dim)
        return s, a, r, s_, self.reward_attributes['gamma']

    def get_best_sequential_actions(self, sess, exp):
        '''
        Used in run_value_estimator_regression_based to get all actions.

        :return best_actions:
            B x (D + 1) action array. The last dimension is just time_pass and needs to be all 1
        '''

        best_actions = self.get_best_actions(sess, exp['cur_state'])
        dummy_time_pass = np.ones((len(best_actions), 1))

        best_actions = np.concatenate((best_actions, dummy_time_pass), axis=1)
        return best_actions
