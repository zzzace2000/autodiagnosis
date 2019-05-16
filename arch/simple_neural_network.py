import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, r2_score

from utils.general import build_nn_module


class MultiClassNNClassifier:
    def __init__(self, variable_scope, input_dim, output_dim,
                 lr, keep_prob, reg_constant, num_hidden_layers,
                 num_hidden_units, normalized_file, log=None, **kwargs):
        self.variable_scope = variable_scope
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lr = lr
        self.keep_prob = keep_prob
        self.reg_constant = reg_constant

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units

        self.learning_step_counter = 0
        self.log = log if log is not None else print
        self.normalized_file = normalized_file

        self._build_model()

    def _init_placeholders(self):
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='output')

    def _init_input_mean_and_std(self):

        if self.normalized_file is None:
            return

        state_mean_and_std = pd.read_csv(self.normalized_file)
        mean, std = state_mean_and_std['mean'].values, state_mean_and_std['std'].values

        assert len(mean) <= self.input_dim
        self.log('Normalize the input for the first %d dim with total %d dim' % (len(mean), self.input_dim))

        if len(mean) < self.input_dim:
            mean = np.concatenate((mean, np.zeros(self.input_dim - len(mean))), axis=0)
            std = np.concatenate((std, np.ones(self.input_dim - len(std))), axis=0)

        self.input_mean = tf.constant(mean[None, :], dtype=tf.float32)
        self.input_std = tf.constant(std[None, :], dtype=tf.float32)

    def _build_net(self):

        inputs = self.input

        if self.normalized_file is not None:
            inputs = (inputs - self.input_mean) / self.input_std

        inputs = build_nn_module(inputs, self.num_hidden_layers, self.num_hidden_units,
                                 self.reg_constant, self.keep_prob, self.is_training)

        outputs = tf.contrib.layers.fully_connected(
            inputs, self.output_dim, activation_fn=None,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))

        return outputs

    def _get_prob(self):
        return tf.nn.sigmoid(self.output)

    def _get_loss(self):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.target))
        tf.summary.scalar(name='loss', tensor=loss)
        return loss

    def _get_reg_loss(self, scope):
        reg_loss = self.reg_constant * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
        tf.summary.scalar(name='reg_loss', tensor=reg_loss)
        return reg_loss

    def _get_train_op(self, scope):
        params = tf.trainable_variables(scope=scope)
        return tf.train.AdamOptimizer(self.lr_node).minimize(self.loss + self.reg_loss, var_list=params)

    def decrease_lr(self, sess):
        """ decrease learning rate when call """
        orig_lr = sess.run(self.lr_node)
        sess.run(self.add_global)
        next_lr = sess.run(self.lr_node)

        self.log('Decrease lr from {} to {}'.format(orig_lr, next_lr))

    def _get_learning_rate(self):
        lr_node = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=1,
                                             decay_rate=0.5, staircase=True)
        tf.summary.scalar('lr', lr_node)
        return lr_node

    def _build_model(self):
        self._init_placeholders()
        self._init_input_mean_and_std()

        with tf.variable_scope('network'):
            self.output = self._build_net()
            self.prob = self._get_prob()

        with tf.variable_scope('lr_decay'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.add_global = self.global_step.assign_add(1)
            self.lr_node = self._get_learning_rate()

        with tf.variable_scope('loss'):
            self.reg_loss = self._get_reg_loss(self.variable_scope + '/network')
            self.loss = self._get_loss()

        with tf.variable_scope('train'):
            self._train_op = self._get_train_op(self.variable_scope + '/network')

        self.summary_ops = tf.summary.merge_all()

    def fit_loader(self, sess, exp_loader, process_exp_func, is_training, batch_print=100,
                   writer=None, debug=False):
        sum_loss, sum_reg_loss, sum_num = 0, 0, 0
        result = {}
        all_probs, all_targets = [], []

        for batch_idx, exp in enumerate(exp_loader):
            batch_size = len(exp[list(exp.keys())[0]])

            input_i, target_i = process_exp_func(exp)

            feed_dict = {self.input: input_i, self.target: target_i, self.is_training: is_training}

            if is_training:
                _, prob, loss, reg_loss, summary_str = sess.run(
                    [self._train_op, self.prob, self.loss, self.reg_loss, self.summary_ops], feed_dict=feed_dict)
                self.learning_step_counter += 1
            else:
                prob, loss, reg_loss, summary_str = sess.run(
                    [self.prob, self.loss, self.reg_loss, self.summary_ops], feed_dict=feed_dict)

            if writer is not None and is_training:
                writer.add_summary(summary_str, self.learning_step_counter)
                writer.flush()

            all_probs.append(prob)
            all_targets.append(target_i)
            sum_loss += loss * batch_size
            sum_reg_loss += reg_loss * batch_size
            sum_num += batch_size

            if batch_idx % batch_print == 0 and batch_idx != 0:
                print('Batch {}: loss {:.3}, reg: {:.3}, auroc: {:.3}'.format(
                    batch_idx, sum_loss / sum_num, sum_reg_loss / sum_num,
                    roc_auc_score(y_true=np.concatenate(all_targets, axis=0),
                                  y_score=np.concatenate(all_probs, axis=0))))

        # Handle writer in validation that only outputs graph / specific loss
        if writer is not None and is_training is False:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=sum_loss * 1.0 / sum_num),
                tf.Summary.Value(tag="reg_loss", simple_value=sum_reg_loss * 1.0 / sum_num),
            ])
            writer.add_summary(summary, self.learning_step_counter)
            writer.flush()

        result.update({'loss'     : sum_loss * 1.0 / sum_num, 'reg_loss': sum_reg_loss * 1.0 / sum_num,
                       'total_num': sum_num,
                       'auroc'    : roc_auc_score(y_true=np.concatenate(all_targets, axis=0),
                                                  y_score=np.concatenate(all_probs, axis=0))})
        return result


class NNRegressor(MultiClassNNClassifier):
    def _get_loss(self):
        batch_size = tf.cast(tf.shape(self.input)[0], tf.float32)
        loss = tf.nn.l2_loss(self.output - self.target) / batch_size
        tf.summary.scalar(name='loss', tensor=loss)
        return loss

    @staticmethod
    def _calculate_pearson_correlation(all_targets, all_outputs):
        assert all_targets.shape[0] == all_outputs.shape[0]
        if all_outputs.ndim == 1:
            all_outputs = all_outputs[:, None]
        if all_targets.ndim == 1:
            all_targets = all_targets[:, None]

        sum_pearson_score = 0.0
        _, output_dim = all_targets.shape

        for i in range(output_dim):
            sum_pearson_score += pearsonr(all_targets[:, i], all_outputs[:, i])[0]
        return sum_pearson_score / output_dim

    @staticmethod
    def _calculate_spearman_correlation(all_targets, all_outputs):
        assert all_targets.shape[0] == all_outputs.shape[0]
        if all_outputs.ndim == 1:
            all_outputs = all_outputs[:, None]
        if all_targets.ndim == 1:
            all_targets = all_targets[:, None]

        sum_score = 0.0
        _, output_dim = all_targets.shape

        for i in range(output_dim):
            score, _ = spearmanr(all_targets[:, i], all_outputs[:, i])
            sum_score += score

        return sum_score / output_dim
    @staticmethod
    def _calculate_r_square(all_targets, all_outputs):
        assert all_targets.shape[0] == all_outputs.shape[0]
        if all_outputs.ndim == 1:
            all_outputs = all_outputs[:, None]
        if all_targets.ndim == 1:
            all_targets = all_targets[:, None]

        return r2_score(y_true=all_targets, y_pred=all_outputs, multioutput='variance_weighted')

    @staticmethod
    def _calculate_mean_and_sd(x):
        return np.mean(x, axis=0, keepdims=True), np.std(x, axis=0, keepdims=True)

    def get_output_i(self, sess, input_i):
        feed_dict = {self.input: input_i, self.is_training: False}
        return sess.run(self.output, feed_dict=feed_dict)

    def get_output(self, sess, exp_loader, process_exp_func):
        all_outputs = []

        for batch_idx, exp in enumerate(exp_loader):
            input_i = process_exp_func(exp)

            output = self.get_output_i(sess=sess, input_i=input_i)

            all_outputs.append(output)

        return np.concatenate(all_outputs, axis=-1)

    def _get_target_and_feed_dict(self, exp, process_exp_func, is_training):
        input_i, target_i = process_exp_func(exp)

        feed_dict = {self.input: input_i, self.target: target_i, self.is_training: is_training}

        return target_i, feed_dict

    def fit_loader(self, sess, exp_loader, process_exp_func, is_training, batch_print=100,
                   writer=None, debug=False):

        all_targets, all_outputs, sum_loss, sum_reg_loss, sum_num = \
            self.fit_loader_raw(sess, exp_loader, process_exp_func, is_training, batch_print,
                                writer, debug)

        pea = self._calculate_pearson_correlation(all_targets=all_targets, all_outputs=all_outputs)
        spearman_corr = self._calculate_spearman_correlation(
            all_targets=all_targets, all_outputs=all_outputs)
        r_square = self._calculate_r_square(all_targets=all_targets, all_outputs=all_outputs)

        # Handle writer in validation that only outputs graph / specific loss
        if writer is not None and is_training is False:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=sum_loss * 1.0 / sum_num),
                tf.Summary.Value(tag="reg_loss", simple_value=sum_reg_loss * 1.0 / sum_num),
                tf.Summary.Value(tag="r_square", simple_value=r_square),
                tf.Summary.Value(tag="spearman",
                                 simple_value=spearman_corr),
                tf.Summary.Value(tag="pea", simple_value=pea),
            ])

            writer.add_summary(summary, self.learning_step_counter)
            writer.flush()

        result = {}
        result.update(dict(loss=sum_loss * 1.0 / sum_num,
                           reg_loss=sum_reg_loss * 1.0 / sum_num,
                           total_num=sum_num, spearman_corr=spearman_corr,
                           r_square=r_square, pea=pea))

        return result

    def fit_loader_raw(self, sess, exp_loader, process_exp_func, is_training, batch_print=100,
                       writer=None, debug=False):
        sum_loss, sum_reg_loss, sum_num = 0, 0, 0
        all_outputs, all_targets = [], []

        for batch_idx, exp in enumerate(exp_loader):
            batch_size = len(exp[list(exp.keys())[0]])

            target_i, feed_dict = self._get_target_and_feed_dict(exp=exp, process_exp_func=process_exp_func,
                                                                 is_training=is_training)

            if is_training:
                _, output_i, loss, reg_loss, summary_str = sess.run(
                    [self._train_op, self.output, self.loss, self.reg_loss, self.summary_ops],
                    feed_dict=feed_dict)
            else:
                output_i, loss, reg_loss, summary_str = sess.run(
                    [self.output, self.loss, self.reg_loss, self.summary_ops], feed_dict=feed_dict)

            all_outputs.append(output_i)
            all_targets.append(target_i)
            sum_loss += loss * batch_size
            sum_reg_loss += reg_loss * batch_size
            sum_num += batch_size

            if writer is not None and is_training:
                writer.add_summary(summary_str, self.learning_step_counter)
                writer.flush()

            if batch_print is not None and batch_idx % batch_print == 0:
                self.log('Batch {}: loss {:.3}, reg: {:.3}'.format(
                    batch_idx, sum_loss / sum_num, sum_reg_loss / sum_num))

            if debug and batch_idx > 100:
                break

        all_targets, all_outputs = np.concatenate(all_targets, axis=0), \
                                   np.concatenate(all_outputs, axis=0)
        return all_targets, all_outputs, sum_loss, sum_reg_loss, sum_num


class KTailNNRegressor(NNRegressor):
    def __init__(self, num_shared_layers, num_sep_layers, **kwargs):
        self.num_shared_layers = num_shared_layers
        self.num_sep_layers = num_sep_layers

        NNRegressor.__init__(self, **dict({'num_hidden_layers': None}, **kwargs))

    def _init_placeholders(self):
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name='input')
        self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='output')
        self.output_mask = tf.placeholder(tf.float32, [None, self.output_dim], name='output_mask')

    def _build_net(self):
        inputs = self.input

        if self.normalized_file is not None:
            inputs = (inputs - self.input_mean) / self.input_std

        with tf.variable_scope('shared_rep'):
            rep = \
                build_nn_module(input=inputs, num_layers=self.num_shared_layers,
                                num_hidden_units=self.num_hidden_units,
                                reg_constant=self.reg_constant,
                                keep_prob=self.keep_prob,
                                is_training=self.is_training)

        with tf.variable_scope('sep_rep'):
            sep_rep = build_nn_module(rep, self.num_sep_layers, self.num_hidden_units,
                                      self.reg_constant, self.keep_prob, self.is_training)

        with tf.variable_scope('out'):
            out = tf.contrib.layers.fully_connected(
                sep_rep, self.output_dim, activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.reg_constant))

            out = out * self.output_mask

        return out

    def _get_target_and_feed_dict(self, exp, process_exp_func, is_training):
        input_i, output_mask_i, target_i = process_exp_func(exp)

        feed_dict = {self.input : input_i, self.output_mask: output_mask_i,
                     self.target: target_i, self.is_training: is_training}

        return target_i, feed_dict

    def _calculate_pearson_correlation(self, all_targets, all_outputs):
        batch_idx = np.arange(all_targets.shape[0])
        output_mask = np.argmax(all_targets != 0, axis=1)
        return pearsonr(all_outputs[batch_idx, output_mask], all_targets[batch_idx, output_mask])[0]

    @staticmethod
    def _calculate_mean_and_sd(x):
        batch_idx = np.arange(x.shape[0])
        output_mask = np.argmax(x != 0, axis=1)
        selected_x = x[batch_idx, output_mask]
        return np.mean(selected_x), np.std(selected_x)


def test_multi_class_classifier():
    def mapping_fun(x):
        return np.stack((np.sum(x * np.array([-0.1, 1, 0.5]), axis=1) > 0,
                         np.sum(x * np.array([0.2, -0.2, 0.3]), axis=1) > 0),
                        axis=1)

    def get_exp_loader(is_train=True):
        np.random.seed(is_train)
        for i in range(5):
            x = np.random.randn(64, 3)
            yield {'input': x, 'target': mapping_fun(x)}

    proc_exp_func = lambda d: (d['input'], d['target'])

    varible_scope = 'simple_network'
    with tf.Session() as sess:
        with tf.variable_scope(varible_scope, reuse=tf.AUTO_REUSE):
            net = MultiClassNNClassifier(variable_scope=varible_scope, input_dim=3, output_dim=2, lr=1e-2,
                                         keep_prob=0.9, reg_constant=1e-4, num_hidden_layers=2, num_hidden_units=32,
                                         log=None)

        sess.run(tf.variables_initializer(tf.global_variables(scope=varible_scope)))

        for i in range(10):
            exp_loader = get_exp_loader(is_train=True)
            net.fit_loader(sess, exp_loader, proc_exp_func, True, batch_print=2)

        exp_loader = get_exp_loader(is_train=False)
        results = net.fit_loader(sess, exp_loader, proc_exp_func, False)
        print('test auroc: {}'.format(results['auroc']))


def test_nn_regressor():
    def mapping_fun(x):
        return np.stack((np.sum(x * np.array([-0.1, 1, 0.5]), axis=1),
                         np.sum(x * np.array([0.2, -0.2, 0.3]), axis=1)),
                        axis=1)

    def get_exp_loader(is_train=True):
        np.random.seed(is_train)
        for i in range(2):
            x = np.random.randn(64, 3)
            yield {'input': x, 'target': mapping_fun(x)}

    proc_exp_func = lambda d: (d['input'], d['target'])

    varible_scope = 'simple_network'
    with tf.Session() as sess:
        with tf.variable_scope(varible_scope, reuse=tf.AUTO_REUSE):
            net = NNRegressor(variable_scope=varible_scope, input_dim=3, output_dim=2, lr=1e-2,
                              keep_prob=0.9, reg_constant=1e-4, num_hidden_layers=2, num_hidden_units=32,
                              normalized_file=None, log=None)

        sess.run(tf.variables_initializer(tf.global_variables(scope=varible_scope)))

        exp_loader = get_exp_loader(is_train=False)
        results = net.fit_loader(sess, exp_loader, proc_exp_func, False)
        print('test r2: {}, spearman: {}, pea: {}'.format(
            results['r_square'], results['spearman_corr'], results['pea']))

        for i in range(1000):
            print(f'Train epoch {i} ---')
            exp_loader = get_exp_loader(is_train=True)
            net.fit_loader(sess, exp_loader, proc_exp_func, True, batch_print=2)

            if i % 10 == 0:
                print(f'Test epoch {i} ---')
                exp_loader = get_exp_loader(is_train=False)
                results = net.fit_loader(sess, exp_loader, proc_exp_func, False)
                print('test r2: {}, corr: {}'.format(results['r_square'],
                                                     results['spearman_corr']))

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    test_nn_regressor()
    # test_multi_class_classifier()
