import tensorflow as tf
from .MGP_RNN import MGP_RNN
from .LMC_MGP_RNN import LMC_MGP_RNN


class ManyToOneDecorator:
    def calculate_predictions_and_probability(self):
        self.last_hidden_state = self.outputs[:, -1, :]
        norm_last_hidden_state = tf.contrib.layers.batch_norm(
            self.last_hidden_state, center=True, scale=True,
            is_training=self.is_training, scope='bn')

        self.raw_preds = tf.matmul(norm_last_hidden_state, self.out_weights) + self.out_biases

        # To get rid of the dependency to self.rnn_inputs through self.N
        N = tf.shape(self.raw_preds)[0]
        self.probs = tf.reshape(tf.nn.softmax(self.raw_preds, -1),
                                [N, self.n_mc_smps, 1, self.n_classes])
        self.avg_probs = tf.reduce_mean(self.probs, axis=1)
        self.O_pred = tf.argmax(self.avg_probs, -1, output_type=tf.int32)

    def fit_loss(self):
        self.O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(self.O, [1, self.n_mc_smps]),
                                                   [self.N * self.n_mc_smps]), depth=self.n_classes)
        self.loss_fit = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.raw_preds,
                                                                               labels=self.O_dupe_onehot) * \
                                       self.target_weights_dupe) / tf.cast(self.N, tf.float32)

        self.loss_reg = self.l2_penalty * tf.reduce_sum(tf.square(self.out_weights))
        for i in range(self.n_layers):
            self.loss_reg += self.l2_penalty * tf.reduce_sum(
                tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_' + str(i) + '/basic_lstm_cell/kernel')))

        self.loss = self.loss_fit + self.loss_reg


class ManyToOneMGP_RNN(ManyToOneDecorator, MGP_RNN):
    def __init__(self, **kwargs):
        kwargs['use_target_weight'] = False
        super(ManyToOneMGP_RNN, self).__init__(**kwargs)


class ManyToOneLMC_MGP_RNN(ManyToOneDecorator, LMC_MGP_RNN):
    def __init__(self, **kwargs):
        kwargs['use_target_weight'] = False
        super(ManyToOneLMC_MGP_RNN, self).__init__(**kwargs)
