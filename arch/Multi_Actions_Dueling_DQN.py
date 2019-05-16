from arch.Dueling_DQN import Dueling_DQN
import tensorflow as tf
import numpy as np


class Multi_Actions_Dueling_DQN(Dueling_DQN):
    '''
    For each state transition, there is at least one actions performed.

    Assume all actions except null action operates on the same time point.
    Assume the last action is null action, account for time change.
    Assume Q values of the combinatorial actions are the linear sum of the Q values of all compositional single actions

    '''
    # def __init__(self, **kwargs):
    #
    #     super(Multi_Actions_Dueling_DQN, self).__init__(**kwargs)

    def __init__(self, q_threshold=0.1, **kwargs):
        self.q_threshold = q_threshold
        super(Multi_Actions_Dueling_DQN, self).__init__(**kwargs)


    def _unwarp_memory(self, batch_memory):
        s = batch_memory[:, :self.num_features]
        a = batch_memory[:, self.num_features : self.num_features + self.num_outputs]

        r = batch_memory[:, self.num_features + self.num_outputs]

        s_ = batch_memory[:, self.num_features + self.num_outputs + 1 : 2 * self.num_features + self.num_outputs + 1]
        a_ = batch_memory[:, -self.num_outputs:] if not self.off_policy else None

        return s, a, r, s_, a_

    def _init_tf_params(self):
        super(Multi_Actions_Dueling_DQN, self)._init_tf_params()
        self.a = tf.placeholder(tf.float32, [None, self.num_outputs], name='one_hot_a')
        self.a_ = tf.placeholder(tf.float32, [None, self.num_outputs], name='one_hot_a_')

    def _get_q_eval(self):
        return tf.reduce_sum(self.q_cur * self.a, axis=1)

    def _get_best_actions_v2(self, q):
        batch_size = tf.shape(q)[0]
        mask_null_acitons = tf.tile(tf.expand_dims(tf.equal(tf.range(self.num_outputs),
                                                            self.num_outputs - 1), [0]),[batch_size, 1])
        mask_true_actions = tf.logical_not(mask_null_acitons)
        pos_true_q = tf.logical_and(q > 0, mask_true_actions)

        best_actions = tf.logical_or(pos_true_q, mask_null_acitons)
        return tf.cast(best_actions, tf.float32)

    def _get_best_actions(self, q):
        batch_size = tf.shape(q)[0]
        mask_null_acitons = tf.tile(tf.expand_dims(tf.equal(tf.range(self.num_outputs),
                                                            self.num_outputs - 1), [0]), [batch_size, 1])
        mask_true_actions = tf.logical_not(mask_null_acitons)
        pos_true_q = tf.logical_and(q > 0, mask_true_actions)

        max_q = tf.reduce_max(q, axis=1, keepdims=True)
        mask_above_threshold = q >= (max_q * self.q_threshold)

        best_actions = tf.logical_or(tf.logical_and(pos_true_q, mask_above_threshold), mask_null_acitons)
        return tf.cast(best_actions, tf.float32)

    def _get_q_target(self):
        a_ = self._get_best_actions(self.q_next) if self.off_policy else self.a_

        return self.r + self.gamma * tf.reduce_sum(self.q_next * a_, axis=1)

    def _get_random_actions(self, batch_size):
        actions = np.random.randint(0, 2, (batch_size, self.num_outputs))
        actions[:, -1] = 1
        return actions


class K_Actions_Dueling_DQN(Multi_Actions_Dueling_DQN):
    '''
    Follow DRRN-Sum and pick max K actions

    He, Ji, et al. “Deep Reinforcement Learning with a Combinatorial Action Space for Predicting Popular Reddit Threads.”
     Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Association for
     Computational Linguistics, 2016, pp. 1838–1848. ACLWeb, https://aclweb.org/anthology/D16-1189.

    '''
    def __init__(self, max_num_actions=5, **kwargs):
        self.max_num_actions = max_num_actions
        super(K_Actions_Dueling_DQN, self).__init__(**kwargs)

    def _get_best_actions(self, q):
        batch_size = tf.shape(q)[0]

        # mask for null action
        mask_null_acitons = tf.tile(tf.expand_dims(tf.equal(tf.range(self.num_outputs),
                                                            self.num_outputs - 1), [0]), [batch_size, 1])
        # mask for positive action
        mask_true_actions = tf.logical_not(mask_null_acitons)
        mask_pos_true_q = tf.logical_and(q > 0, mask_true_actions)

        # mask for top-k q value
        top_k_val, top_k_idx = tf.nn.top_k(q, self.max_num_actions, sorted=False)
        my_range = tf.expand_dims(tf.range(0, batch_size), 1)
        my_range_repeated = tf.tile(my_range, [1, self.max_num_actions])
        full_indices = tf.concat(axis=2, values=[tf.expand_dims(my_range_repeated, 2), tf.expand_dims(top_k_idx, 2)])
        full_indices = tf.reshape(full_indices, [-1, 2])
        mask_top_k = tf.not_equal(tf.sparse_to_dense(full_indices, tf.shape(q), tf.reshape(top_k_val, [-1]),
                                        default_value=0., validate_indices=False), 0.)

        # combining mask
        best_actions = tf.logical_or(tf.logical_and(mask_pos_true_q, mask_top_k), mask_null_acitons)

        return tf.cast(best_actions, tf.float32)
