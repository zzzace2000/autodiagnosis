import numpy as np
import tensorflow as tf
from arch.memory import Memory
from utils.general import build_nn_module
from arch.simple_neural_network import simple_neural_network

class Dueling_DQN(simple_neural_network):
    def __init__(
            self,
            variable_scope,
            num_features,
            num_outputs,
            lr=1e-3,
            keep_prob=0.5,
            reg_constant=1e-3,
            dropout_on=False, # determine whether or not using stochistic or deterministic policy
            off_policy=True,
            gamma=0.9,
            e_greedy=0.9,
            e_greedy_increment=0.0,

            num_hidden_joint_layers=2,
            num_hidden_sep_layers=2,
            num_hidden_joint_units=32,
            num_hidden_sep_units=64,

            batch_size=32,
            replace_target_iter=200,
            memory_size=500,
            logfile=None
    ):

        # Model Arch -----
        self.num_hidden_joint_layers = num_hidden_joint_layers
        self.num_hidden_joint_units = num_hidden_joint_units
        self.num_hidden_sep_layers = num_hidden_sep_layers
        self.num_hidden_sep_units = num_hidden_sep_units

        # learning params ---------
        self.batch_size = batch_size

        # classifier_training_and_evaluation params
        self.off_policy = off_policy
        self.gamma = gamma
        self.keep_prob = keep_prob
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment else self.epsilon_max

        # dqn specific learning params ---------
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.memory = Memory(capacity=memory_size)  # Experience replay
        self.memory_counter = 0

        super(Dueling_DQN, self).__init__(variable_scope=variable_scope,
                                          num_features=num_features,
                                          num_outputs=num_outputs,
                                          num_hidden_layers=None,
                                          num_hidden_units=None,
                                          lr=lr,
                                          keep_prob=keep_prob,
                                          reg_constant=reg_constant,
                                          dropout_on=dropout_on, # todo: support mc dropout
                                          logfile=logfile)

    def _init_tf_params(self):
        self.s = tf.placeholder(tf.float32, [None, self.num_features], name='s')
        self.a = tf.placeholder(tf.float32, [None], name='a')
        self.r = tf.placeholder(tf.float32, [None], name='r')
        self.s_ = tf.placeholder(tf.float32, [None, self.num_features], name='s_')
        self.a_ = tf.placeholder(tf.float32, [None], name='a_')

        self.ISWeights = tf.placeholder(tf.float32, [None], name='ISWeights')

        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.add_global = self.global_step.assign_add(1)
        self.lr_node = tf.train.exponential_decay(self.lr, global_step=self.global_step,
                                                  decay_steps=1, decay_rate=0.5, staircase=True)
    def _build_net(self, input_features):
        with tf.variable_scope('representation'):
            input_features = build_nn_module(input_features, self.num_hidden_joint_layers, self.num_hidden_joint_units,
                                              self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('value'):
            value = build_nn_module(input_features, self.num_hidden_sep_layers, self.num_hidden_sep_units,
                                    self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('advantage'):
            adv = build_nn_module(input_features, self.num_hidden_sep_layers, self.num_hidden_sep_units,
                                        self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('q'):
            value = tf.contrib.layers.fully_connected(
                value, 1, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))
            adv = tf.contrib.layers.fully_connected(
                adv, self.num_outputs, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))
            q = value + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))  # Q = V(s) + A(s,a)

        return q

    def _unwarp_memory(self, batch_memory):
        s = batch_memory[:, :self.num_features]
        a = batch_memory[:, self.num_features]
        r = batch_memory[:, self.num_features + 1]
        s_ = batch_memory[:, self.num_features + 2: 2 * self.num_features + 2]
        a_ = batch_memory[:, -1] if not self.off_policy else None

        return s, a, r, s_, a_

    def _get_q_eval(self):
        batch_idx = tf.range(tf.shape[self.s[0]])
        q_eval = tf.gather_nd(self.q_cur, tf.stack((batch_idx, tf.cast(self.a, tf.int32)), axis=1))

        return q_eval

    def _get_q_target(self):

        if self.off_policy:
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1)
        else:
            batch_idx = tf.range(tf.shape[self.s[0]])
            q_target = self.r + self.gamma * tf.gather_nd(self.q_next, tf.stack((batch_idx, self.a_), axis=1))

        return q_target

    def _get_best_actions(self, q):
        return tf.argmax(q, axis=1)

    def _get_loss(self):
        batch_size = tf.cast(tf.shape(self.q_target)[0], tf.float32)
        return tf.nn.l2_loss(self.ISWeights * (self.q_target - self.q_eval)) / batch_size

    def _get_replace_target_op(self):
        t_params = tf.trainable_variables(scope=self.variable_scope + '/target_net')
        e_params = tf.trainable_variables(scope=self.variable_scope + '/eval_net')

        # Update target net params ------------
        return [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_model(self):
        # Initialize tf params ------------
        self._init_tf_params()

        # build evaluate_net --------------
        with tf.variable_scope('eval_net'):
            self.q_cur = self._build_net(self.s)

        # Get best action based on Current Q -------
        self.best_actions = self._get_best_actions(self.q_cur)

        # build target_net ------------
        with tf.variable_scope('target_net'):
            self.q_next = self._build_net(self.s_)

        # Update target net params ------------
        self.replace_target_op = self._get_replace_target_op()

        # Calculate loss -----------------------
        with tf.variable_scope('loss'):
            self.q_eval = self._get_q_eval()
            self.q_target = self._get_q_target()

            self.abs_errors = tf.abs(self.q_target - self.q_eval)         # for updating SumTree
            self.reg_loss = self._get_reg_loss(scope=self.variable_scope + '/eval_net')
            self.loss = self._get_loss()

        # Set up training operation -------------
        with tf.variable_scope('train'):
            self._train_op = self._get_train_op(scope=self.variable_scope + '/eval_net')


    def store_transition(self, s, a, r, s_, a_=None):
        '''
        Store current state, action, reward and next state to memory buffer
        '''

        batch_size = s.shape[0]

        if len(r.shape) == 1:
            r = r[:, None]

        if self.off_policy:
            transitions = np.hstack((s, a, r, s_))
        else:
            assert a_ is not None
            transitions = np.hstack((s, a, r, s_, a_))

        for i in range(batch_size):
            self.memory.store(transitions[i,:])

        self.memory_counter += batch_size

    def _get_random_actions(self, batch_size):
        return np.random.randint(0, self.num_outputs, batch_size)

    def choose_action(self, sess, s):
        random_try = np.random.uniform() > self.epsilon

        if random_try:
            actions = self._get_random_actions(batch_size=s.shape[0])
        else:
            actions = sess.run(self.best_actions, feed_dict={self.s: s})

        return actions

    def run(self, sess, is_training, s=None, a=None, r=None, s_=None, a_=None, verbose=False):
        tree_idx = None

        if is_training:
            batch_size = self.batch_size
            if self.learning_step_counter % self.replace_target_iter == 0:
                sess.run(self.replace_target_op)

            tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
            s, a, r, s_, a_ = self._unwarp_memory(batch_memory)

        else:
            batch_size = s.shape[0]
            ISWeights = np.ones(batch_size)

        assert s is not None and a is not None and r is not None and s_ is not None
        # Evaluate TD error square loss ---------
        feed_dict = {self.s: s,
                     self.a: a,
                     self.r: r,
                     self.s_: s_,
                     self.ISWeights: ISWeights,
                     self.is_training: is_training}

        if not self.off_policy:
            assert a_ is not None
            feed_dict[self.a_] = a_

        if is_training:
            _, abs_errors, loss, reg_loss = sess.run([self._train_op, self.abs_errors, self.loss, self.reg_loss],
                                                     feed_dict=feed_dict)

            self.memory.batch_update(tree_idx, abs_errors)  # update priority
            self.epsilon = self.epsilon + self.epsilon_increment \
                if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learning_step_counter += 1

        else:
            loss, reg_loss = sess.run([self.loss, self.reg_loss], feed_dict=feed_dict)


        result = {'loss': loss, 'reg_loss': reg_loss}
        self._reporter(verbose, result)

        return result #loss, reg_loss
