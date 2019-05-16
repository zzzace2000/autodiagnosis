import os, json
import tensorflow as tf
import numpy as np

from database.MIMIC_cache_exp import MIMIC_cache_discretized_joint_exp_random_order, \
    MIMIC_cache_discretized_exp_env_v3, MIMIC_cache_discretized_exp_env_v3_last_obs
from arch.simple_neural_network import NNRegressor, KTailNNRegressor
from arch.RNN import ManyToOneRNN


def load_model(sess, model_dir, rnn_dir=None):
    hyper_path = os.path.join(model_dir, 'hyperparameters.json')
    if not os.path.exists(hyper_path):
        raise FileNotFoundError('No file found in the path %s' % hyper_path)

    params = json.load(open(hyper_path, 'r'))

    cls = eval(params['model_type'])
    with tf.variable_scope(cls.variable_scope, reuse=tf.AUTO_REUSE):
        model = cls(**params)

    # Best model idx
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.variable_scope))
    if os.path.exists(os.path.join(model_dir, 'results.json')):
        best_model_idx = json.load(open(os.path.join(model_dir, 'results.json')))['best_model_idx']
        print('Find results.json ... The best model idx is %d' % best_model_idx)
        saver.restore(sess, os.path.join(model_dir, 'model-%d' % best_model_idx))
    else:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    if rnn_dir is not None and not isinstance(model, ProbGainEstimatorMixin):
        rnn = ManyToOneRNN.load_saved_mgp_rnn(sess, rnn_dir)
        model.rnn = rnn

    return model


def base_decorator(Cls):
    ''' Add an attribute of cache_dir to produce data_loader '''
    def __init__(self, cache_dir, **kwargs):
        self.cache_dir = cache_dir
        kwargs['variable_scope'] = self.variable_scope

        self.rnn = None
        if 'rnn' in kwargs:
            self.rnn = kwargs.pop('rnn')
        if 'mimic_cache_cls' in kwargs and kwargs['mimic_cache_cls'] is not None:
            self.mimic_cache_cls = kwargs.pop('mimic_cache_cls')
        else:
            self.mimic_cache_cls = 'MIMIC_cache_discretized_exp_env_v3'

        super(Cls, self).__init__(**kwargs)

    def get_predicted_prob_gain(self, sess, exp, agent_a):
        '''
        Default is the estimator directly estimates prob_gain. Override this func when needed
        :return: Return a (B,) shape of np array indicating prob_gain for these actions
        '''
        raise NotImplementedError('Specify this function!')

    Cls.__init__ = __init__
    Cls.get_predicted_prob_gain = get_predicted_prob_gain
    return Cls

@base_decorator
class BaseNNRegressor(NNRegressor):
    pass

@base_decorator
class BaseKTailNNRegressor(KTailNNRegressor):
    pass


class PerStepDataLoaderMixin:
    def get_data_loader(self, mode, sess, batch_size):
        mimic_exp = MIMIC_cache_discretized_joint_exp_random_order(cache_dir=self.cache_dir)
        if mode == 'train':
            return mimic_exp.gen_train_experience(sess, batch_size=batch_size, shuffle=True)
        elif mode == 'val':
            return mimic_exp.gen_val_experience(sess, batch_size=batch_size, shuffle=False)

        # Per-step: test-time just uses a one_pass thing
        return mimic_exp.gen_experience(sess=sess, filename='test_onepass_all',
                                        batch_size=batch_size, shuffle=False)


class PerTimeDataLoaderMixin:
    def get_data_loader(self, mode, sess, batch_size):
        mimic_cache_cls = eval(self.mimic_cache_cls)
        mimic_exp = mimic_cache_cls(cache_dir=self.cache_dir)
        if mode == 'train':
            return mimic_exp.gen_experience('train_per_time_env', sess, batch_size=batch_size, shuffle=True)
        elif mode == 'val':
            return mimic_exp.gen_experience('val_per_time_env', sess, batch_size=batch_size, shuffle=False)

        return mimic_exp.gen_experience('test_per_time_env', sess, batch_size=batch_size, shuffle=False)

    def _get_tuple(self, exp):
        return


class ProbGainEstimatorMixin:
    def get_predicted_prob_gain(self, sess, exp, agent_a=None, data_dict=None):
        inputs, _ = self.process_exp_func(exp, agent_a)
        return self.get_output_i(sess=sess, input_i=inputs)[:, 0]


class StateDiffEstimatorMixin:
    def get_predicted_prob_gain(self, sess, exp, agent_a=None, data_dict=None):
        assert self.rnn is not None, 'You forget to initialize by passing to the rnn'

        inputs, _ = self.process_exp_func(exp, agent_a)
        output = self.get_output_i(sess=sess, input_i=inputs)
        return self.get_predicted_prob_gain_from_output(sess, exp, output)

    def get_predicted_prob_gain_from_output(self, sess, exp, output):
        pred_next_state = self._post_process_func(exp, output)

        # The dimension originally is B x n_mc x 1 x n_cls
        pred_prob = sess.run(self.rnn.probs, feed_dict={
            self.rnn.last_hidden_state: pred_next_state,
            self.rnn.is_training: False,
        })[:, 0, 0, 1]

        return pred_prob - exp['cur_prob']

    def _post_process_func(self, exp, output):
        return exp['cur_state'] + output


class StateEstimatorMixin(StateDiffEstimatorMixin):
    def _post_process_func(self, exp, output):
        return output


class ObsDiffEstimatorMixin:
    def get_pred_next_obs(self, sess, exp, agent_a):
        assert self.rnn is not None, 'You forget to initialize by passing to the rnn'

        inputs, _ = self.process_exp_func(exp, agent_a)

        obs_diff = self.get_output_i(sess=sess, input_i=inputs)
        pred_next_obs = exp['cur_obs'] + obs_diff
        return pred_next_obs

    def get_predicted_prob_gain(self, sess, exp, agent_a=None, data_dict=None):
        assert data_dict is not None, 'you should pass in data_dict!'
        pred_next_obs = self.get_pred_next_obs(sess, exp, agent_a)

        # TODO: how to handle obs?


class ObsEstimatorMixin(ObsDiffEstimatorMixin):
    def get_pred_next_obs(self, sess, exp, agent_a):
        assert self.rnn is not None, 'You forget to initialize by passing to the rnn'

        inputs, _ = self.process_exp_func(exp, agent_a)
        pred_next_obs = self.get_output_i(sess=sess, input_i=inputs)
        return pred_next_obs


class StateToProbGainPerStepEstimator(ProbGainEstimatorMixin,
                                      PerStepDataLoaderMixin, BaseNNRegressor):
    ''' type = 1 '''
    variable_scope = 'reward_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 207
        kwargs['output_dim'] = 1
        super(StateToProbGainPerStepEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['actions']
        inputs = np.concatenate([exp['cur_state'], exp['cur_history'], agent_a], axis=1)
        targets = exp['prob_gain'][:, None]
        return inputs, targets


class StateToProbGainKTailsPerStepEstimator(ProbGainEstimatorMixin,
                                            PerStepDataLoaderMixin, BaseKTailNNRegressor):
    ''' type = 2 '''
    variable_scope = 'reward_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 39
        super(StateToProbGainKTailsPerStepEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['actions']
        inputs = np.concatenate([exp['cur_state'], exp['cur_history']], axis=1)
        output_mask = agent_a
        targets = output_mask * exp['prob_gain'][:, None]
        return inputs, output_mask, targets


class StateToStateDiffPerStepEstimator(StateDiffEstimatorMixin,
                                       PerStepDataLoaderMixin, BaseNNRegressor):
    ''' type = 3 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 207
        kwargs['output_dim'] = 128
        super(StateToStateDiffPerStepEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['actions']
        inputs = np.concatenate([exp['cur_state'], exp['cur_history'], agent_a], axis=1)
        targets = exp['next_state'] - exp['cur_state']
        return inputs, targets


class StateToStatePerTimeEstimator(StateEstimatorMixin,
                                   PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 4 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 128
        super(StateToStatePerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_state'], agent_a], axis=1)
        targets = exp['next_state']
        return inputs, targets


class StateToStateDiffPerTimeEstimator(StateDiffEstimatorMixin,
                                       PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 4 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 128
        super(StateToStateDiffPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_state'], agent_a], axis=1)
        targets = exp['next_state'] - exp['cur_state']
        return inputs, targets


class StateToProbGainPerTimeEstimator(ProbGainEstimatorMixin,
                                      PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 5 '''
    variable_scope = 'reward_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 1
        super(StateToProbGainPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_state'], agent_a], axis=1)
        targets = exp['prob_gain'][:, None]
        return inputs, targets


class StateToObsDiffPerTimeEstimator(ObsDiffEstimatorMixin,
                                     PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 6 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 39
        super(StateToObsDiffPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_state'], agent_a], axis=1)
        targets = exp['next_obs'] - exp['cur_obs']
        return inputs, targets


class StateToObsPerTimeEstimator(ObsEstimatorMixin,
                                 PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = ? '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 167
        kwargs['output_dim'] = 39
        super(StateToObsPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_state'], agent_a], axis=1)
        targets = exp['next_obs']
        return inputs, targets


class ObsToObsDiffPerTimeEstimator(ObsDiffEstimatorMixin,
                                   PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 7 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 78
        kwargs['output_dim'] = 39
        super(ObsToObsDiffPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_obs'], agent_a], axis=1)
        targets = exp['next_obs'] - exp['cur_obs']
        return inputs, targets


class ObsToObsPerTimeEstimator(ObsEstimatorMixin,
                               PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 8 '''
    variable_scope = 'transition_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 78
        kwargs['output_dim'] = 39
        super(ObsToObsPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_obs'], agent_a], axis=1)
        targets = exp['next_obs']
        return inputs, targets


class ObsToProbGainPerTimeEstimator(ProbGainEstimatorMixin,
                                    PerTimeDataLoaderMixin, BaseNNRegressor):
    ''' type = 9 '''
    variable_scope = 'reward_model'

    def __init__(self, **kwargs):
        kwargs['input_dim'] = 78
        kwargs['output_dim'] = 1
        super(ObsToProbGainPerTimeEstimator, self).__init__(**kwargs)

    def process_exp_func(self, exp, agent_a=None):
        if agent_a is None:
            agent_a = exp['cur_actions']
        inputs = np.concatenate([exp['cur_obs'], agent_a], axis=1)
        targets = exp['prob_gain'][:, None]
        return inputs, targets
