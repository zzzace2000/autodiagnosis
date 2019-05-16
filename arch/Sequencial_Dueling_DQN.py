import tensorflow as tf

from arch.K_Tails_Dueling_DQN import KTailsDuelingDQN
from utils.general import build_nn_module
import numpy as np


class SequencialDuelingDQN(KTailsDuelingDQN):
    """
    Sequentially decide which action to be made and whether or not to skip to the next time point
    """
    def _init_placeholders(self):
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.is_weights = tf.placeholder(tf.float32, [None], name='is_weights')  # weights for prioritized exp replay
        self.gamma = tf.placeholder(tf.float32, shape=[None], name='gamma')

        self.s = tf.placeholder(tf.float32, [None, self.state_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.state_dim], name='s_')
        self.a = tf.placeholder(tf.int32, [None, self.action_dim], name='a')
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')

    def _get_history_action(self, s):
        pad_zero = tf.zeros(shape=[tf.shape(self.s)[0], 1])
        raw_history_action = s[:, -(self.action_dim - 1):]
        return tf.concat([raw_history_action, pad_zero], axis=1)

    def _unwarp_memory(self, batch_memory):
        s = batch_memory[:, :self.state_dim]
        a = batch_memory[:, self.state_dim: self.state_dim + self.action_dim]
        r = batch_memory[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim + 1]
        s_ = batch_memory[:, self.state_dim + self.action_dim + 1:]

        return s, a, r, s_

    def _build_net(self, obs):
        with tf.variable_scope('shared_rep'):
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

        with tf.variable_scope('q'):  # Q(s_t, a_t) = V(s_t) + A(s_t, a_t)
            value = tf.contrib.layers.fully_connected(
                value, 1, activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))  # batch_size x 1

            adv = tf.contrib.layers.fully_connected(
                adv, self.action_dim, activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))  # batch_size x action_dim

            q = value + (adv - tf.reduce_mean(adv, axis=-1, keepdims=True))  # batch_size x action_dim

        return q

    def _select_q(self, q, selected_a):
        """
        Return the corresponding q value for the action choice
        q: batch_size x action_dim
        selected_a: batch_size
        """
        batch_size = tf.shape(self.s)[0]
        batch_idx = tf.range(batch_size)
        selected_q = tf.gather_nd(q, tf.stack((batch_idx, selected_a), axis=-1))
        return tf.reshape(selected_q, shape=[-1, 1])  # batch_size x 1

    def _get_q_eval(self):
        selected_a = tf.argmax(self.a, axis=1, output_type=tf.int32)
        return self._select_q(q=self.q_cur, selected_a=selected_a)

    def _adjust_q_value(self, q):
        """
        Return adjusted q value such that any actions that have been take will have a lower q value than the other
        """
        cur_history_action = self._get_history_action(self.s)  # batch_size x action_dim
        min_q = tf.reduce_min(q, axis=-1, keepdims=True)  # batch_size x 1

        # Set the value of history action as (min-1)
        adjusted_q = cur_history_action * (min_q - 1) + (1 - cur_history_action) * q
        return adjusted_q

    def _get_q_target(self):
        """
        Return q target: r + gamma * Q'(s_{t+1}, a_{t+1}}
        Added hard constraint on which next step's action can be made
        """
        adjusted_q = self._adjust_q_value(self.q_next)
        selected_a = tf.argmax(adjusted_q, axis=-1, output_type=tf.int32)
        selected_q = self._select_q(q=adjusted_q, selected_a=selected_a)

        return self.r + tf.expand_dims(self.gamma, 1) * selected_q

    # def _get_q_target(self):
    #     return self.r + self.gamma * tf.reduce_max(self.q_next, axis=-1)  # batch_size x action_dim

    def get_best_sequential_actions(self, sess, exp):
        cur_state = exp['cur_state']
        return self._get_best_sequential_actions(sess, cur_state)

    def _get_best_sequential_actions(self, sess, cur_state):
        batch_size = cur_state.shape[0]

        agent_a = np.zeros((batch_size, self.action_dim))

        # Start with empty history
        cur_history = np.zeros((batch_size, self.action_dim - 1))
        kept = np.ones((batch_size,), dtype=bool)

        # The maximum is the 40 dimensions all performed!
        for k in range(self.action_dim):
            if np.all(agent_a[:, -1] == 1):  # All records are time pass (False)
                break

            s = np.concatenate((cur_state, cur_history), axis=1)

            best_action = self.get_best_actions(sess=sess, s=s)
            agent_a[kept, best_action[kept]] = 1

            # If action is time shift, flag it as non-kept
            kept = (kept & (best_action != self.action_dim - 1))

            cur_history[kept, best_action[kept]] = 1

        # Sanity check
        assert np.all(agent_a[:, -1] == 1), \
            'Not all action is time_pass:\n' + str(agent_a[:, -1])
        return agent_a

    def _get_best_actions(self):
        """
        Return best action for current state
        Added hard constraint on which next step's action can be made
        """
        adjusted_q = self._adjust_q_value(self.q_cur)
        return tf.argmax(adjusted_q, axis=-1)

    def plot_tb_best_action_dist(self, sess, exp_loader, writer, topK=3, image=False, identifier=''):
        '''
        Plot the top K actions histogram.
        '''
        best_action_result = []
        num_actions_till_time_pass_result = []

        for batch_idx, exp in enumerate(exp_loader):
            batch_size = len(exp['cur_state'])

            # Record best actions and corresponding len
            best_action_arrs = -1 * np.ones((batch_size, topK))
            num_actions_till_time_pass = topK * np.ones((batch_size,))

            # Start with empty history
            cur_history = np.zeros((batch_size, self.action_dim - 1))
            kept = np.ones((batch_size,), dtype=bool)

            for k in range(topK):
                if not np.any(kept): # All records are time pass
                    break

                s = np.concatenate((exp['cur_state'], cur_history), axis=1)

                best_action = self.get_best_actions(sess=sess, s=s)
                best_action_arrs[kept, k] = best_action[kept]

                # Update num_actions! Last are kept but now becomes time pass
                num_actions_till_time_pass[kept & (best_action == self.action_dim - 1)] = (k+1)

                # If action is time shift, flag it as non-kept
                kept = (kept & (best_action != self.action_dim - 1))

                cur_history[kept, best_action[kept]] = 1

            best_action_result.append(best_action_arrs)
            num_actions_till_time_pass_result.append(num_actions_till_time_pass)

        best_action_result = np.concatenate(best_action_result, axis=0)
        num_actions_till_time_pass_result = np.concatenate(num_actions_till_time_pass_result, axis=0)

        def _get_summary(values, tag, max_val):
            if image:
                return self._get_tf_image_summary_with_histogram(sess, values, tag, max_val)
            return self._get_tf_histogram_summary(values, tag, max_val)

        for k in range(topK):
            # Remove the -1
            values = best_action_result[:, k][best_action_result[:, k] != -1]
            if len(values) == 0:
                continue
            summary = _get_summary(values, tag='%sactions_top%d' % (identifier, k),
                                   max_val=self.action_dim+1)
            writer.add_summary(summary, self.learning_step_counter)

        values = best_action_result[best_action_result != -1]
        if len(values) > 0:
            summary = _get_summary(values, tag='%sactions_joint_top' % (identifier),
                                   max_val=self.action_dim+1)
            writer.add_summary(summary, self.learning_step_counter)

        # Record
        summary = _get_summary(num_actions_till_time_pass_result,
                               tag='%snum_actions_till_time_pass' % (identifier),
                               max_val=np.max(num_actions_till_time_pass_result) + 2)
        writer.add_summary(summary, self.learning_step_counter)

        writer.flush()

    def _reward_func(self, exp):
        # If mortality is 1, then encourage to increase probability.
        # If mortality is 0, be negative.
        information_gain = self.reward_attributes['gain_coef'] * exp['prob_gain'] * (2 * exp['labels'] - 1)

        # action cost. Time pass cost is 0, others are 1
        time_pass = exp['actions'][:, -1]
        action_cost = 1 - time_pass

        rewards = information_gain - self.reward_attributes['action_cost_coef'] * action_cost
        return rewards[:, None]  # Increase one dim

    def _get_gamma(self, a):
        time_pass = a[:, -1] == 1
        return time_pass * self.reward_attributes['gamma'] + (1 - time_pass)

    def process_exp_func(self, exp):
        r = self._reward_func(exp)

        # If it's time pass, then use original state. If it's not, use delay_update_state
        time_pass = exp['actions'][:, -1:]
        cur_state = exp['delay_update_state']
        next_state = time_pass * exp['next_state'] + (1 - time_pass) * exp['delay_update_state']

        s = np.concatenate((cur_state, exp['cur_history']), axis=-1)
        s_ = np.concatenate((next_state, exp['next_history']), axis=-1)
        a = exp['actions']
        return s, a, r, s_, self._get_gamma(exp['actions'])


class LastObsSequencialDuelingDQN(SequencialDuelingDQN):
    '''
    Use the last observation as input to the DQN instead of hidden state
    '''
    def __init__(self, **kwargs):
        # No normalization for using last observation
        kwargs['normalized_file'] = None
        super(LastObsSequencialDuelingDQN, self).__init__(**kwargs)

    def process_exp_func(self, exp):
        r = self._reward_func(exp)

        # If it's time pass, then use original state. If it's not, use delay_update_state
        time_pass = exp['actions'][:, -1:]
        cur_state = time_pass * exp['t_2_obs'] + (1 - time_pass) * exp['t_1_obs']
        next_state = exp['t_1_obs']

        s = np.concatenate((cur_state, exp['cur_history']), axis=-1)
        s_ = np.concatenate((next_state, exp['next_history']), axis=-1)
        a = exp['actions']
        return s, a, r, s_, self._get_gamma(exp['actions'])

    def get_best_sequential_actions(self, sess, exp):
        '''
        Based on the environement models, it's t_1_obs.
        '''
        cur_state = exp['t_1_obs']
        return self._get_best_sequential_actions(sess, cur_state)
