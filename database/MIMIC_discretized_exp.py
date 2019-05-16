import os

import numpy as np

if __name__ == '__main__':
    import sys

    sys.path.append('../')

from utils.math import cross_entropy_loss
from database.MIMIC_window import MIMIC_discretized_window
from arch.RNN import ManyToOneRNN, ManyToOneMGP_RNN
import copy
from database.Exceptions import InvalidPatientException
import json
import pandas as pd
from database.MIMIC_cache_exp import MIMIC_cache_discretized_joint_exp_random_order, \
    MIMIC_cache_discretized_joint_exp_independent_measurement, \
    MIMIC_cache_discretized_joint_exp_random_order_with_obs


class MIMIC_discretized_exp(MIMIC_discretized_window):
    '''
    Do an independent experience for each k action.
    The s_(t+1) is generated from each corresponding action a_t
    Prove to be wrong in the simulation.

    Corresponding decache database: MIMIC_cache_discretized_exp
    '''

    def __init__(self, reward_func, mgp_rnn, RL_interval, min_hours_of_patient=0, **kwargs):
        self.mgp_rnn = mgp_rnn

        self.reward_func = reward_func
        self.RL_interval = RL_interval
        self.min_hours_of_patient = min_hours_of_patient

        # Here we let RNN interval (data_interval) and
        # discretization interval (dis_interval) the same as RL interval.
        # So everything is discretized into grid point for RL to generate experience
        kwargs['data_interval'] = RL_interval
        kwargs['dis_interval'] = RL_interval

        # When caching, be verbose
        kwargs['verbose'] = 1

        super(MIMIC_discretized_exp, self).__init__(**kwargs)

    def _is_valid(self, Y, T, ind_kf, ind_kt, label, episode_end_time):
        start_ind, end_ind = super(MIMIC_discretized_exp, self)._is_valid(
            Y, T, ind_kf, ind_kt, label, episode_end_time)

        num_hours_warmup = max(T[ind_kt[self.min_measurements_in_warmup]], self.num_hours_warmup)
        if episode_end_time < self.min_hours_of_patient + num_hours_warmup + self.before_end:
            raise InvalidPatientException(
                'Patient only stays %.1f hours but min is %.1f exp and %.1f warmup, %.1f before end with label %d' %
                (episode_end_time, self.min_hours_of_patient, num_hours_warmup, self.before_end, label))

        return start_ind, end_ind

    def cache_experience(self, sess, identifier, batch_size=None,
                         RL_exp_dir='../RL_exp_cache/', mode='testing',
                         cache_proportion=1., cache_type='all'):
        the_idxes = self._get_idxes(cache_type, cache_proportion, mode)
        self._cache_experience(
            sess, identifier, the_idxes, batch_size=batch_size, name=mode, RL_exp_dir=RL_exp_dir)

    def gen_pat_data_loader(self, mode, batch_size, catch_type='all', cache_proportion=1.):
        the_idxes = self._get_idxes(mode=mode, cache_type=catch_type, cache_proportion=cache_proportion)
        data_loader = self._gen_pat_data_loader(batch_size=batch_size, patient_idxes=the_idxes)
        return data_loader

    def _get_idxes(self, cache_type, cache_proportion, mode):
        if cache_type == 'all':
            train_idxes = self.train_idxes
            val_idxes = self.val_idxes
            test_idxes = self.test_idxes
        elif cache_type == 'pos':
            train_idxes = self.pos_train_idxes
            val_idxes = self.pos_val_idxes
            test_idxes = self.pos_test_idxes
        elif cache_type == 'neg_sampled':
            def subsample(pos, neg):
                neg_sampled = np.random.choice(neg, len(pos), replace=False)
                return np.random.permutation(np.concatenate((pos, neg_sampled), axis=0))

            train_idxes = subsample(self.pos_train_idxes, self.neg_train_idxes)
            val_idxes = subsample(self.pos_val_idxes, self.neg_val_idxes)
            # Still take full idxes of test set. But random order is not suitable for caching...
            test_idxes = self.test_idxes
        else:
            raise Exception('No this cache type....')

        if cache_proportion < 1.:
            def subset(idxes):
                return np.random.choice(idxes, int(len(idxes) * cache_proportion), replace=False)

            train_idxes = subset(train_idxes)
            val_idxes = subset(val_idxes)
            test_idxes = subset(test_idxes)

        if mode == 'testing':
            return [0, 1]
        if mode == 'train':
            return train_idxes
        elif mode == 'val':
            return val_idxes
        elif mode == 'test' or mode == 'test_onepass_all':
            return test_idxes
        else:
            raise Exception('No mode like this %s' % mode)

    def _cache_experience(self, sess, identifier, idxes,
                          name, batch_size=None, RL_exp_dir='../RL_exp_cache/',
                          gen_reward_func=None):
        out_dir = os.path.join(RL_exp_dir, identifier)
        out_fname = os.path.join(out_dir, '%s.tfrecords' % name)

        if not os.path.exists(RL_exp_dir):
            os.mkdir(RL_exp_dir)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if os.path.isfile(out_fname):
            os.remove(out_fname)

        writer = tf.python_io.TFRecordWriter(out_fname)

        if gen_reward_func is None:
            gen_reward_func = self._gen_rewards_input
        rewards_input = gen_reward_func(sess, batch_size, idxes)

        num_examples_cache = 0
        for batch, r_input in enumerate(rewards_input):
            for i in range(len(r_input['labels'])):
                # Create a feature
                feature = self._construct_cache_feature(i, r_input)

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            num_examples_cache += len(r_input['labels'])

            if batch % 50 == 0:
                print('batch %d' % batch, 'num_examples_cache:', num_examples_cache)

        writer.close()
        sys.stdout.flush()

        print('%s data: total num examples cache:' % name, num_examples_cache)

        # Return meta data to
        result = {'num_examples_cache': num_examples_cache, 'num_patients': len(idxes)}
        json.dump(result, open(os.path.join(RL_exp_dir, identifier, '%s_stats.json' % name), 'w'))

        return num_examples_cache

    def _construct_cache_feature(self, i, r_input):
        # Save some space. But can not be less than 32 for some reasons...
        cur_state = r_input['cur_states'][i].astype(np.float32)
        next_state = r_input['next_states'][i].astype(np.float32)
        gain = r_input['gain'][i].astype(np.float32)

        feature = {
            'cur_state': self._bytes_feature(cur_state),
            'next_state': self._bytes_feature(next_state),
            'gain': self._bytes_feature(gain),
            'label': self._int64_feature(r_input['labels'][i]),
            'patient_inds': self._int64_feature(r_input['patient_inds'][i]),
            'the_steps': self._int64_feature(r_input['the_steps'][i]),
            'total_steps': self._int64_feature(r_input['total_steps'][i]),
            'action': self._bytes_feature(r_input['actions'][i]),
            'mortality': self._int64_feature(r_input['mortality'][i]),
        }

        self._handle_specific_feature(feature, i, r_input)
        return feature

    def _handle_specific_feature(self, feature, i, r_input):
        pass

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        str_rep = tf.compat.as_bytes(value.tostring())
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str_rep]))

    def _gen_rewards_input(self, sess, batch_size, patient_idxes):
        '''
        Generate all the input to be cached. Include s_t, s_(t+1) and gain etc.
        :return: the input to the self._construct_cache_feature
        '''
        loader = self._gen_pat_data_loader(
            batch_size=batch_size, patient_idxes=patient_idxes)

        total_patients = next(loader)
        self._verbose_print('Total patients %d' % total_patients)

        for the_data_dict in loader:
            result = self._get_states_and_info_gains(sess, the_data_dict)

            # Put the rest into the result
            for k in ['labels', 'patient_inds', 'the_steps', 'total_steps', 'mortality']:
                result[k] = the_data_dict[k]

            yield result

    def _get_states_and_info_gains(self, sess, data_dict):
        '''
        Generate s_t and information gain on (t+1). Run 3 times of RNN:
        - s_t: derived from RNN hidden state by running till ending at time t
        - info_t: derived from RNN output loss in (t+1) w/ no measurement in t+1
        - info_(t+1): derived from RNN output loss in (t+1) w/ indep. measurement in t+1
        :param data_dict: output from data loader. Affected by self._get_ind_arr()
        :return: s_t, s_(t+1), info_gain (info_(t+1) - info_t)
        '''

        all_obs_dict, actions = self._preprocess_all_obs_dicts(data_dict)

        # Get the cross entropy loss
        state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob = \
            self._gen_states_and_rnn_output(sess, all_obs_dict)

        return self._cal_states_and_info_gains(
            state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob, actions)

    def _preprocess_all_obs_dicts(self, data_dict):
        '''
        Calculate in data_dict how many indices are actions (action_time = pred time X[-1])
        :return current_obs_dict, no_action_dict, action_dict, actions
        '''
        one_hot_actions = self._get_one_hot_actions(
            data_dict['num_actions'], data_dict['ind_kfs'])

        current_obs_dict = {
            'Ys': [Y[:len(Y) - num_action] for Y, num_action in zip(data_dict['Ys'], data_dict['num_actions'])],
            'ind_kfs': [ind_kf[:len(ind_kf) - num_action]
                        for ind_kf, num_action in zip(data_dict['ind_kfs'], data_dict['num_actions'])],
            'ind_kts': [ind_kt[:len(ind_kt) - num_action]
                        for ind_kt, num_action in zip(data_dict['ind_kts'], data_dict['num_actions'])],
            'Ts': data_dict['Ts'],
            'labels': data_dict['labels'],  # labels are that time point's label
            # Move back an interval to t
            'Xs': [X - self.RL_interval for X in data_dict['Xs'] if X - self.RL_interval >= 0],
            'covs': data_dict['covs'],
        }

        # 'Xs': [X - (T[ind_kt[-1]] - T[ind_kt[len(ind_kt) - (num_action + 1)]]) for X, ind_kt, T, num_action in
        #        zip(data_dict['Xs'], data_dict['ind_kts'], data_dict['Ts'], data_dict['num_actions'])],

        # Get info_no_action
        no_action_dict = copy.deepcopy(current_obs_dict)
        no_action_dict['Xs'] = data_dict['Xs']

        # All obs dict concatenated
        all_obs_dict = {}
        for k in current_obs_dict.keys():
            all_obs_dict[k] = current_obs_dict[k] + no_action_dict[k] + data_dict[k]

        return all_obs_dict, one_hot_actions

    def _get_one_hot_actions(self, num_actions, ind_kfs):
        one_hot_actions = np.zeros((len(num_actions), self.num_features), dtype=np.uint8)

        for idx, (ind_kf, num_action) in enumerate(zip(ind_kfs, num_actions)):
            actions = ind_kf[(len(ind_kf) - num_action):]
            for a in actions:
                one_hot_actions[idx, a] = 1

        return one_hot_actions

    def _cal_states_and_info_gains(self, state_combo, mean_ce_loss, std_ce_loss,
                                   mean_prob, std_prob, actions):
        num = state_combo.shape[0] // 3

        cur_states = state_combo[:num, ...]
        next_states = state_combo[-num:, ...]

        gain = mean_ce_loss[num:(2 * num)] - mean_ce_loss[-num:]
        gain_std = np.sqrt(std_ce_loss[num:(2 * num)] ** 2 + std_ce_loss ** 2)

        cur_prob = mean_prob[num:(2 * num)]
        next_prob = mean_prob[-num:]

        cur_prob_std = std_prob[num:(2 * num)]
        next_prob_std = std_prob[-num:]

        return dict(cur_states=cur_states, next_states=next_states, gain=gain, gain_std=gain_std,
                    cur_prob=cur_prob, next_prob=next_prob, cur_prob_std=cur_prob_std,
                    next_prob_std=next_prob_std, actions=actions)

    def _gen_states_and_rnn_output(self, sess, obs_dict):
        pad_dict = self.mgp_rnn.pad_measurements(**obs_dict)
        feed_dict = self.mgp_rnn.create_feed_dict(**pad_dict)

        state, probs = sess.run([
            self.mgp_rnn.states, self.mgp_rnn.probs], feed_dict=feed_dict)

        # Take RNN last states mean and variance as classifier_training_and_evaluation state
        state = state[-1][-1].reshape(-1, self.mgp_rnn.n_mc_smps, self.mgp_rnn.n_hidden)
        state_mean = np.mean(state, axis=1)
        state_std = np.std(state, axis=1)
        state_combo = np.concatenate((state_mean, state_std), axis=1)

        # calculate Cross entropy loss
        # labels = np.tile(np.concatenate(obs_dict['labels']), 3)[:, None]
        probs = probs[:, :, 0, 1]
        ce_loss = cross_entropy_loss(obs_dict['labels'], probs)

        mean_ce_loss = ce_loss.mean(axis=-1)
        std_ce_loss = ce_loss.std(axis=-1)

        mean_prob = probs.mean(axis=-1)
        std_prob = probs.std(axis=-1)

        return state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob

    def _record_results_by_idxes(self, ind_arr_result, Y, T, ind_kt, ind_kf,
                                 label, episode_end_time, result_dict, patient_idx,
                                 cov=None):
        ''' Hack: Override to include mortality label & current time index array '''
        super(MIMIC_discretized_exp, self)._record_results_by_idxes(
            ind_arr_result, Y, T, ind_kt, ind_kf, label, episode_end_time,
            result_dict, patient_idx, cov)

        if 'num_actions' not in result_dict:
            result_dict['num_actions'] = []
        if 'mortality' not in result_dict:
            result_dict['mortality'] = []

        result_dict['num_actions'].append(ind_arr_result['num_action'])
        result_dict['mortality'].append(label)

        # Include number of actions that is in (t-1)
        if 'num_prev_time_action' in ind_arr_result:
            if 'num_prev_time_actions' not in result_dict:
                result_dict['num_prev_time_actions'] = []
            result_dict['num_prev_time_actions'].append(ind_arr_result['num_prev_time_action'])

        # Include number of actions that is in (t-2)
        if 'num_t_2_time_action' in ind_arr_result:
            if 'num_t_2_time_actions' not in result_dict:
                result_dict['num_t_2_time_actions'] = []
            result_dict['num_t_2_time_actions'].append(ind_arr_result['num_t_2_time_action'])

    def _gen_ind_arr_loader(self, start_ind, end_ind, episode_end_time, T, ind_kt):
        '''
        Hack: I override the index array generation to get the current state and next state.
        2 conditions:
        <1> No measurement for this RL interval. Then generate an idx array with last element as None.
            I handle this special case in the base class.
        <2> Having measurements for this interval. Then generate a series of index array with each
            measurement as an independent action.
        Call the two functions (_gen_ind_arr_for_empty_action, _gen_ind_arr_for_action)
        '''
        current_ind = next_ind = start_ind
        first_ind_arr = list(range(start_ind))

        start_time = max(episode_end_time - self.include_before_death,
                         T[ind_kt[self.min_measurements_in_warmup]], self.num_hours_warmup)
        end_time = episode_end_time - self.before_end

        def transform_time_to_idx(time):
            return int((time - (T[0] - self.dis_interval / 2 + 0.001)) // self.dis_interval)

        start_idx = transform_time_to_idx(start_time)
        end_idx = transform_time_to_idx(end_time)

        the_arr = range(start_idx + 1, end_idx + 1)
        total_step = len(the_arr)

        # Record t-1 and t-2 how many actions: to record history for sequential updates
        num_prev_time_action, num_t_2_time_action = 0, 0

        for step, the_ind_t in enumerate(the_arr):
            # Subset array to forget previous measurements to stablize gp
            if len(first_ind_arr) > self.num_gp_samples_limit:
                first_ind_arr = first_ind_arr[(len(first_ind_arr) - self.num_gp_samples_limit):]

            while next_ind <= end_ind and ind_kt[next_ind] == the_ind_t:
                next_ind += 1

            # If next index is still the same, that means it could not find any measurement
            # for the_ind_t, so it just generates an empty second_ind_arr
            second_ind_arr = list(range(current_ind, next_ind))

            all_ind_arrs = self._gen_ind_tuples_for_action(first_ind_arr, second_ind_arr, num_prev_time_action,
                                                           num_t_2_time_action, step)
            for the_dict in all_ind_arrs:
                assert 'ind_arr' in the_dict and 'num_action' in the_dict
                the_dict.update(dict(end_pred_time=T[the_ind_t], step=step, total_step=total_step))
                yield the_dict

            first_ind_arr += second_ind_arr
            current_ind = next_ind
            num_t_2_time_action = num_prev_time_action
            num_prev_time_action = len(second_ind_arr)

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Generate an array of tuple [(all_indexes, current_time_idxes), ...].
        The "current_time_idxes" are used to indicate which measurements are in the current time.
        '''
        if len(second_ind_arr) == 0:
            return [dict(ind_arr=first_ind_arr, num_action=0)]

        # Generate independent action
        result = []
        for sec_ind in second_ind_arr:
            result.append(dict(ind_arr=first_ind_arr + [sec_ind], num_action=1))
        return result


class MIMIC_discretized_joint_exp(MIMIC_discretized_exp):
    '''
    For each time with k measurements, we generate a joint k action
    experience with all the measurements. Note here the s_(t+1) is the
    corresponding s_(t+1) for each independent action a_(t,i).

    Corresponding decache database: MIMIC_cache_discretized_exp_joint
    '''

    def _handle_specific_feature(self, feature, i, r_input):
        for k in ['cur_prob', 'next_prob']:
            feature[k] = self._float_feature(r_input[k][i])

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Generate an array of tuple [(all_indexes, current_time_idxes), ...].
        The "current_time_idxes" are used to indicate which measurements are in the current time.
        Generate joint experiences
        '''
        return [dict(ind_arr=first_ind_arr + second_ind_arr, num_action=len(second_ind_arr))]


class MIMIC_discretized_joint_exp_independent_measurement(MIMIC_discretized_exp):
    '''
    Return experience with a decrease-in-CE-loss vector (GV)
        Each column of GE represents the gain in CE loss due to the measurement alone subtracting
        the gain in CE loss due to time passing (i.e. NULL action)

        Geneerate independent action but next state is induced by joint actions.
        Reward is a vector with each corresponding action.
        Proved to be correct in simulation.

    Experience returned no longer contain prediction probability
    '''

    def __init__(self, **kwargs):
        # Only generate one patient trajectory at a time
        kwargs['num_pat_produced'] = 1
        super(MIMIC_discretized_joint_exp_independent_measurement, self).__init__(**kwargs)

    def _construct_cache_feature(self, i, r_input):
        feature = {
            'cur_state': self._bytes_feature(r_input['cur_state'][i].astype(np.float32)),
            'next_state': self._bytes_feature(r_input['next_state'][i].astype(np.float32)),

            'cur_action': self._bytes_feature(r_input['cur_action'][i].astype(np.uint8)),
            'next_action': self._bytes_feature(r_input['next_action'][i].astype(np.uint8)),

            'gain_per_action': self._bytes_feature(r_input['gain_per_action'][i].astype(np.float32)),
            'prob_gain_per_action': self._bytes_feature(r_input['prob_gain_per_action'][i].astype(np.float32)),
            'std_gain_per_action': self._bytes_feature(r_input['std_gain_per_action'][i].astype(np.float32)),

            'gain_joint': self._float_feature(r_input['gain_joint'][i]),
            'prob_joint': self._float_feature(r_input['prob_joint'][i]),
            'std_joint': self._float_feature(r_input['std_joint'][i]),
            'prob_null': self._float_feature(r_input['prob_null'][i]),
            'std_null': self._float_feature(r_input['std_null'][i]),
            'prev_prob_joint': self._float_feature(r_input['prev_prob_joint'][i]),
            'prev_std_joint': self._float_feature(r_input['prev_std_joint'][i]),

            'labels': self._int64_feature(r_input['labels'][i]),
            'patient_inds': self._int64_feature(r_input['patient_inds'][i]),
            'the_steps': self._int64_feature(r_input['the_steps'][i]),
            'total_steps': self._int64_feature(r_input['total_steps'][i]),
            'mortality': self._int64_feature(r_input['mortality'][i]),
        }
        return feature

    def _gen_rewards_input(self, sess, batch_size, patient_idxes):
        '''

        Generate (s_t, a_t, s_(t+1), r_t)

        Terminal state have no next actions
        Actions are one hot encoding, potentially multi-hot
        '''

        # Generate a patient whole trajectory one patient at a time!
        loader = self._gen_pat_data_loader(batch_size=None, patient_idxes=patient_idxes)

        total_patients = next(loader)
        self._verbose_print('Total patients %d' % total_patients)

        for patient_obs_dict in loader:
            per_action_patient_obs_dict, action_arr_per_time_point = self._gen_per_action_obs_dict(patient_obs_dict)

            result = self._process_splited_obs_dict(sess, per_action_patient_obs_dict, action_arr_per_time_point)

            for k in ['labels', 'patient_inds', 'the_steps', 'total_steps', 'mortality']:
                result[k] = patient_obs_dict[k]

            yield result

    def _gen_per_action_obs_dict(self, obs_dict: dict):
        '''
        Generate obs dict with all joint measurements split up
        for each exp, if new measurement are made at the next exp, then
            num(new measurements) + 2 will be added to a new obs_dict
            But, when num(new measurements) == 0 or 1, we only add (num(new measurements) + 1) exp

        for 1st exp, we need to additionally generate t-1 state w/ joint action.
        Then for every exp, we generate (K + 2) obs_dict.
        The order is (empty action -- individual action (K) -- joint action)
        '''
        sparse_mat_keys = ['Ys', 'ind_kfs', 'ind_kts']  # splitted up exp are differed only by the measurements
        other_mat_keys = ['Ts', 'Xs', 'labels', 'patient_inds', 'the_steps', 'total_steps', 'covs']

        new_obs_dict = {k: [] for k in sparse_mat_keys + other_mat_keys}

        # For each time point
        for i, num_action in enumerate(obs_dict['num_actions']):
            for k in new_obs_dict:
                if k in sparse_mat_keys:
                    # Insert empty action exp
                    new_obs_dict[k].append(obs_dict[k][i][:len(obs_dict[k][i]) - num_action])
                    # Insert independent action exp (K)
                    for a_idx in range(len(obs_dict[k][i]) - num_action, len(obs_dict[k][i])):
                        new_obs_dict[k].append(obs_dict[k][i][:len(obs_dict[k][i]) - num_action]
                                               + [obs_dict[k][i][a_idx]])

                    # Insert all actions
                    if num_action > 1:  # Save some computation
                        new_obs_dict[k].append(obs_dict[k][i])
                else:
                    offset = num_action if num_action > 1 else num_action - 1
                    for _ in range(2 + offset):
                        new_obs_dict[k].append(obs_dict[k][i])

        # Handle special case: <1> insert 1st experience w/ t-1 array.
        # Copy 1st experience but shift back X to -self.RL_interval
        for k in new_obs_dict:
            new_obs_dict[k].insert(0, new_obs_dict[k][0])
        new_obs_dict['Xs'][0] = [x - self.RL_interval for x in new_obs_dict['Xs'][0]
                                 if x - self.RL_interval > 0]

        # Generate action. It follows the order in the array
        action_arr_per_time_point = [
            ind_kf[len(ind_kf) - num_action:] for num_action, ind_kf in zip(
                obs_dict['num_actions'], obs_dict['ind_kfs'])]

        return new_obs_dict, action_arr_per_time_point

    def _process_splited_obs_dict(self, sess, obs_dict, action_arr_per_time_point):

        state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob = \
            self._gen_states_and_rnn_output(sess, obs_dict)

        information_gain_per_action, prob_gain_per_action, std_gain_per_action = \
            (np.zeros((len(action_arr_per_time_point), self.num_features)) for _ in range(3))

        information_gain_joint, prob_null, std_null = \
            (np.zeros(len(action_arr_per_time_point)) for _ in range(3))

        # Actions & states have cur_action / next_action. The final action is empty
        one_hot_actions = np.zeros((len(action_arr_per_time_point) + 1, self.num_features))
        states = np.zeros((len(action_arr_per_time_point) + 1, state_combo.shape[1]))
        prob_joints, std_joints = (np.zeros(len(action_arr_per_time_point) + 1) for _ in range(2))

        prev_idx = 0  # idx exp w/ joint action at t-1
        states[-1, :] = state_combo[-1, :]  # Fill in the last state
        prob_joints[0], std_joints[0] = mean_prob[0], std_prob[0]

        for i, action_arr in enumerate(action_arr_per_time_point):
            # Fill in t-1 state
            states[i, :] = state_combo[prev_idx]

            # Current probability. It's the t w/ empty action
            prob_null[i] = mean_prob[prev_idx + 1]
            std_null[i] = std_prob[prev_idx + 1]

            num_action = len(action_arr)
            offset = num_action if num_action > 1 else num_action - 1

            # Handle joint differences
            information_gain_joint[i] = \
                -mean_ce_loss[prev_idx + offset + 2] + mean_ce_loss[prev_idx + 1]
            prob_joints[1 + i] = mean_prob[prev_idx + offset + 2]
            std_joints[1 + i] = std_prob[prev_idx + offset + 2]

            # Fill in per action gain
            for a_idx, action_id in enumerate(action_arr):
                one_hot_actions[i, action_id] = 1

                # Info gain = ce_loss_t_empty - ce_loss_t_joint
                information_gain_per_action[i, action_id] = \
                    mean_ce_loss[prev_idx + 1] - mean_ce_loss[prev_idx + 2 + a_idx]
                # Prob gain = prob_t_joint - prob_t_empty
                prob_gain_per_action[i, action_id] = \
                    mean_prob[prev_idx + 2 + a_idx] - mean_prob[prev_idx + 1]
                std_gain_per_action[i, action_id] = \
                    std_prob[prev_idx + 2 + a_idx] - std_prob[prev_idx + 1]

            prev_idx += offset + 2  # jump to next t of joint action

        prob_joint, prev_prob_joint = prob_joints[1:], prob_joints[:-1]
        std_joint, prev_std_joint = std_joints[1:], std_joints[:-1]

        result = dict(
            gain_per_action=information_gain_per_action,
            gain_joint=information_gain_joint,
            prob_joint=prob_joint, prob_gain_per_action=prob_gain_per_action,
            prev_prob_joint=prev_prob_joint, prev_std_joint=prev_std_joint,
            std_gain_per_action=std_gain_per_action,
            std_joint=std_joint, prob_null=prob_null, std_null=std_null)

        result['cur_action'], result['next_action'] = one_hot_actions[:-1, :], one_hot_actions[1:, :]
        result['cur_state'], result['next_state'] = states[:-1, :], states[1:, :]
        return result

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Generate an array of tuple [(all_indexes, current_time_idxes), ...].
        The "current_time_idxes" are used to indicate which measurements are in the current time.
        Generate joint experiences
        '''
        return [dict(ind_arr=first_ind_arr + second_ind_arr, num_action=len(second_ind_arr))]


def MGP_RNN_to_RNN_decorator(Cls):
    def _gen_states_and_rnn_output(self, *args, **kwargs):
        ''' Remove the stdev part in the state '''
        state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob = \
            super(Cls, self)._gen_states_and_rnn_output(*args, **kwargs)
        num = state_combo.shape[1] // 2
        return state_combo[:, :num], mean_ce_loss, std_ce_loss, mean_prob, std_prob

    Cls._gen_states_and_rnn_output = _gen_states_and_rnn_output
    return Cls


@MGP_RNN_to_RNN_decorator
class MIMIC_discretized_joint_exp_independent_measurement_rnn(MIMIC_discretized_joint_exp_independent_measurement):
    pass


class MIMIC_discretized_joint_exp_random_order(MIMIC_discretized_exp):
    '''
    Use complicated state space to handle multiple action combination.
    We randomly sample M order to approximate the 2^K possible combinations
    '''

    def __init__(self, *args, num_random_order=30, **kwargs):
        '''
        :param num_random_order: Max random order. If -1, then just 1 pass
        '''
        super(MIMIC_discretized_joint_exp_random_order, self).__init__(*args, **kwargs)
        self.num_random_order = num_random_order

    def _get_states_and_info_gains(self, sess, data_dict):
        '''
        Override to produce normally
        '''
        all_obs_dict = self._preprocess_all_obs_dicts(data_dict)

        return self._get_resulting_features(sess, data_dict, all_obs_dict)

    def _preprocess_all_obs_dicts(self, data_dict):
        '''
        Calculate in data_dict how many indices are actions (action_time = pred time X[-1])
        :return current_obs_dict, no_action_dict, action_dict, actions
        '''
        current_obs_dict = copy.deepcopy(data_dict)
        for idx, num_action in enumerate(data_dict['num_actions']):
            if num_action > 0:
                current_obs_dict['Ys'][idx] = current_obs_dict['Ys'][idx][:-1]
                current_obs_dict['ind_kfs'][idx] = current_obs_dict['ind_kfs'][idx][:-1]
                current_obs_dict['ind_kts'][idx] = current_obs_dict['ind_kts'][idx][:-1]
            else:  # Move from prev time to current time
                X = current_obs_dict['Xs'][idx]
                X = np.array(X) - self.RL_interval
                current_obs_dict['Xs'][idx] = X[X > 0]

        # Calculate original array that has no new measurement at this time point!
        cur_time_no_measurement = copy.deepcopy(data_dict)
        for idx, num_action in enumerate(cur_time_no_measurement['num_actions']):
            if num_action > 0:
                cur_time_no_measurement['Ys'][idx] = cur_time_no_measurement['Ys'][idx][:-num_action]
                cur_time_no_measurement['ind_kfs'][idx] = cur_time_no_measurement['ind_kfs'][idx][:-num_action]
                cur_time_no_measurement['ind_kts'][idx] = cur_time_no_measurement['ind_kts'][idx][:-num_action]
            else:  # Move from prev time to current time
                X = cur_time_no_measurement['Xs'][idx]
                X = np.array(X) - self.RL_interval
                cur_time_no_measurement['Xs'][idx] = X[X > 0]

        # All obs dict concatenated
        all_obs_dict = {}
        for k in current_obs_dict.keys():
            if isinstance(current_obs_dict[k], list):
                all_obs_dict[k] = current_obs_dict[k] + data_dict[k] + cur_time_no_measurement[k]
            elif isinstance(current_obs_dict[k], np.ndarray):
                all_obs_dict[k] = np.concatenate(
                    (current_obs_dict[k], data_dict[k], cur_time_no_measurement[k]), axis=0)
            else:
                raise Exception('Wierd type! ' + str(type(current_obs_dict[k])))

        return all_obs_dict

    def _get_resulting_features(self, sess, data_dict, all_obs_dict):
        # Get the cross entropy loss
        state_combo, mean_ce_loss, std_ce_loss, mean_prob, std_prob = \
            self._gen_states_and_rnn_output(sess, all_obs_dict)

        num = state_combo.shape[0] // 3
        assert len(data_dict['num_actions']) == num, '%d %d' % (len(data_dict['num_actions']), num)

        cur_states = state_combo[:num, ...]
        next_states = state_combo[num:(2 * num), ...]
        delay_update_state = state_combo[(2 * num):, ...]

        cur_prob = mean_prob[:num]
        next_prob = mean_prob[num:(2 * num)]
        prob_gain = next_prob - cur_prob

        cur_history = np.zeros((len(data_dict['num_actions']), self.num_features),
                               dtype=np.uint8)
        next_history = np.zeros((len(data_dict['num_actions']), self.num_features),
                                dtype=np.uint8)
        one_hot_actions = np.zeros((len(data_dict['num_actions']), self.num_features + 1),
                                   dtype=np.uint8)

        for idx, (ind_kf, num_action) in enumerate(
                zip(data_dict['ind_kfs'], data_dict['num_actions'])):
            if num_action <= 0:
                tmp = ind_kf[len(ind_kf) + num_action:]
                for a in tmp:
                    cur_history[idx, a] = 1

                one_hot_actions[idx, -1] = 1
                continue

            tmp = ind_kf[-num_action:]
            for a in tmp[:-1]:
                cur_history[idx, a] = 1
                next_history[idx, a] = 1

            # The action is the last element
            next_history[idx, ind_kf[-1]] = 1
            one_hot_actions[idx, ind_kf[-1]] = 1

        return dict(cur_states=cur_states, next_states=next_states, cur_history=cur_history,
                    next_history=next_history, prob_gain=prob_gain, cur_prob=cur_prob,
                    next_prob=next_prob, actions=one_hot_actions,
                    delay_update_state=delay_update_state)

    def _construct_cache_feature(self, i, r_input):
        # Save some space. But can not be less than 32 for some reasons...
        cur_state = r_input['cur_states'][i].astype(np.float32)
        next_state = r_input['next_states'][i].astype(np.float32)
        delay_update_state = r_input['delay_update_state'][i].astype(np.float32)
        prob_gain = r_input['prob_gain'][i].astype(np.float32)
        cur_prob = r_input['cur_prob'][i].astype(np.float32)
        next_prob = r_input['next_prob'][i].astype(np.float32)

        feature = {
            'cur_state': self._bytes_feature(cur_state),
            'next_state': self._bytes_feature(next_state),
            'delay_update_state': self._bytes_feature(delay_update_state),
            'cur_history': self._bytes_feature(r_input['cur_history'][i]),
            'next_history': self._bytes_feature(r_input['next_history'][i]),
            'actions': self._bytes_feature(r_input['actions'][i]),
            'prob_gain': self._float_feature(prob_gain),
            'cur_prob': self._float_feature(cur_prob),
            'next_prob': self._float_feature(next_prob),
            'labels': self._int64_feature(r_input['labels'][i]),
            'patient_inds': self._int64_feature(r_input['patient_inds'][i]),
            'the_steps': self._int64_feature(r_input['the_steps'][i]),
            'total_steps': self._int64_feature(r_input['total_steps'][i]),
            'mortality': self._int64_feature(r_input['mortality'][i]),
        }

        return feature

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Generate an array of tuple [(all_indexes, num_action), ...].
        The num_action encodes several meaning:
        - If it's 0 or negative, it means the at t-1 how many action is performed
        - If it's positive, it means how many action in all_indexes happen at this time
        '''
        prev_time_transition = [] if step == 0 else \
            [dict(ind_arr=first_ind_arr, num_action=-num_prev_time_action,
                  num_prev_time_action=num_prev_time_action,
                  num_t_2_time_action=num_t_2_time_action)]

        # Simple case
        if len(second_ind_arr) == 0:
            return prev_time_transition
        if len(second_ind_arr) == 1:
            return prev_time_transition + [dict(ind_arr=first_ind_arr + second_ind_arr, num_action=1,
                                                num_prev_time_action=num_prev_time_action,
                                                num_t_2_time_action=num_t_2_time_action)]

        # Just a one pass!
        results = prev_time_transition
        if self.num_random_order == -1:
            permutation = np.random.permutation(second_ind_arr)

            order_ind_arr = []
            for the_ind in permutation:
                order_ind_arr.append(the_ind)
                results.append(dict(ind_arr=first_ind_arr + order_ind_arr, num_action=len(order_ind_arr),
                                    num_prev_time_action=num_prev_time_action,
                                    num_t_2_time_action=num_t_2_time_action
                                    ))
            return results

        # If not one pass, then do a random order up to specified number
        exp_produced = set()

        results = prev_time_transition
        for _ in range(50):
            if len(exp_produced) > self.num_random_order:
                break

            permutation = np.random.permutation(second_ind_arr)

            order_ind_arr = []
            for the_ind in permutation:
                order_ind_arr.append(the_ind)

                key = str((set(order_ind_arr), the_ind))
                if key in exp_produced:
                    continue
                exp_produced.add(key)
                results.append(dict(ind_arr=first_ind_arr + order_ind_arr, num_action=len(order_ind_arr),
                                    num_prev_time_action=num_prev_time_action,
                                    num_t_2_time_action=num_t_2_time_action))

        return results


@MGP_RNN_to_RNN_decorator
class MIMIC_discretized_joint_exp_random_order_rnn(MIMIC_discretized_joint_exp_random_order):
    pass


class LastObsMixin:
    '''
    Additionally cache the observations of (t-2) and (t-1) for every experience.
    Each observation is a concatenation of [obs, covariates, missingness]
    with total 39 + 38 + 39 = 116 dimension
    '''
    def _get_resulting_features(self, sess, data_dict, all_obs_dict):
        features = super(LastObsMixin, self) \
            ._get_resulting_features(sess, data_dict, all_obs_dict)

        # t-1 and t-2 obs
        B = len(data_dict['num_actions'])
        t_1_obs, t_1_missingness = np.zeros((B, self.num_features), dtype=np.float32), \
                                   np.ones((B, self.num_features), dtype=np.float32)
        t_2_obs, t_2_missingness = np.zeros((B, self.num_features), dtype=np.float32), \
                                   np.ones((B, self.num_features), dtype=np.float32)

        for idx, (num_action, num_prev_time_action, num_t_2_time_action, Y, ind_kf) in enumerate(zip(*[
            data_dict[k] for k in ['num_actions', 'num_prev_time_actions', 'num_t_2_time_actions', 'Ys', 'ind_kfs']
        ])):
            the_ind_kf, the_Y = ind_kf, Y
            if num_action > 0: # Remove intermediate step observations
                the_ind_kf = the_ind_kf[:(-num_action)]
                the_Y = the_Y[:(-num_action)]

            for ind_f, y in zip(the_ind_kf[(len(the_ind_kf) - num_prev_time_action):],
                                the_Y[(len(the_Y) - num_prev_time_action):]):
                t_1_obs[idx, ind_f] = y
                t_1_missingness[idx, ind_f] = 0
            for ind_f, y in zip(
                    the_ind_kf[len(the_ind_kf) - (num_t_2_time_action + num_prev_time_action):(len(the_ind_kf)-num_prev_time_action)],
                    the_Y[len(the_Y) - (num_t_2_time_action + num_prev_time_action):(len(the_Y)-num_prev_time_action)]):
                t_2_obs[idx, ind_f] = y
                t_2_missingness[idx, ind_f] = 0

        covs = np.vstack(data_dict['covs'])
        features['t_1_obs'] = np.concatenate((t_1_obs, covs, t_1_missingness), axis=1)
        features['t_2_obs'] = np.concatenate((t_2_obs, covs, t_2_missingness), axis=1)

        return features

    def _construct_cache_feature(self, i, r_input):
        feature = super(LastObsMixin, self) \
            ._construct_cache_feature(i, r_input)

        feature['t_1_obs'] = self._bytes_feature(r_input['t_1_obs'][i].astype(np.float32))
        feature['t_2_obs'] = self._bytes_feature(r_input['t_2_obs'][i].astype(np.float32))
        return feature


class MIMIC_discretized_joint_exp_random_order_with_obs(LastObsMixin, MIMIC_discretized_joint_exp_random_order):
    '''
    Add an observation which is the last time input to the LSTM.
    It concatenates the time's observation, the missingness and the covariates
    '''
    pass


@MGP_RNN_to_RNN_decorator
class MIMIC_discretized_joint_exp_random_order_with_obs_rnn(MIMIC_discretized_joint_exp_random_order_with_obs):
    pass


class MIMIC_per_time_exp(MIMIC_discretized_joint_exp_random_order):
    ''' Cache per time state to plot for the sequentialDQN.plot_tb_best_action_dist. '''

    def _get_idxes(self, cache_type, cache_proportion, mode):
        ''' Make sure when we cache, we cache all the data we have '''
        print('Cache type fixed to all the examples!')
        return super(MIMIC_per_time_exp, self)._get_idxes(
            cache_type='all', cache_proportion=1., mode=mode
        )

    def _cache_experience(self, *args, **kwargs):
        ''' Change name '''
        assert 'name' in kwargs
        kwargs['name'] += '_per_time'
        return super(MIMIC_per_time_exp, self)._cache_experience(*args, **kwargs)

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Return only the experience from prev time to cur time, and take next state as per time step.
        '''
        prev_time_transition = [dict(ind_arr=first_ind_arr, num_action=0)] if step == 0 \
            else [dict(ind_arr=first_ind_arr, num_action=-num_prev_time_action)]
        return prev_time_transition


@MGP_RNN_to_RNN_decorator
class MIMIC_per_time_exp_rnn(MIMIC_per_time_exp):
    pass


class MIMIC_per_time_env_exp(MIMIC_discretized_joint_exp_random_order):
    '''
    Do a per time env experience caching.
    '''

    def _gen_ind_tuples_for_action(self, first_ind_arr, second_ind_arr, num_prev_time_action,
                                   num_t_2_time_action, step):
        '''
        Return the transition that moves from t-1 to t
        '''
        the_ind_arr = first_ind_arr + second_ind_arr
        return [] if step == 0 else [dict(ind_arr=the_ind_arr, num_action=len(second_ind_arr),
                                          num_prev_time_action=num_prev_time_action,
                                          num_t_2_time_action=num_t_2_time_action)]

    def _get_states_and_info_gains(self, sess, data_dict):
        all_obs_dict = self._preprocess_all_obs_dicts(data_dict)

        return self._get_resulting_features(sess, data_dict, all_obs_dict)

    def _preprocess_all_obs_dicts(self, data_dict):
        '''
        Generate a t-1 and t concatenation metrics
        :return current_obs_dict
        '''

        def remove_last_elements(arr, num_removed):
            return arr[:(len(arr) - num_removed)]

        # Generate t-1 dictionary
        current_obs_dict = copy.deepcopy(data_dict)
        for idx, num_action in enumerate(data_dict['num_actions']):
            if num_action > 0:
                current_obs_dict['Ys'][idx] = remove_last_elements(
                    current_obs_dict['Ys'][idx], num_removed=num_action)
                current_obs_dict['ind_kfs'][idx] = remove_last_elements(
                    current_obs_dict['ind_kfs'][idx], num_removed=num_action)
                current_obs_dict['ind_kts'][idx] = remove_last_elements(
                    current_obs_dict['ind_kts'][idx], num_removed=num_action)

            X = current_obs_dict['Xs'][idx]
            X = np.array(X) - self.RL_interval
            current_obs_dict['Xs'][idx] = X[X > 0]

        # Concatenate t-1 and t index array
        all_obs_dict = {}
        for k in current_obs_dict.keys():
            if isinstance(current_obs_dict[k], list):
                all_obs_dict[k] = current_obs_dict[k] + data_dict[k]
            elif isinstance(current_obs_dict[k], np.ndarray):
                all_obs_dict[k] = np.concatenate(
                    (current_obs_dict[k], data_dict[k]), axis=0)
            else:
                raise Exception('Wierd type! ' + str(type(current_obs_dict[k])))

        return all_obs_dict

    def _get_resulting_features(self, sess, data_dict, all_obs_dict):
        # Get the cross entropy loss
        state_combo, _, _, mean_prob, _ = \
            self._gen_states_and_rnn_output(sess, all_obs_dict)

        num = int(state_combo.shape[0] / 2)
        cur_states = state_combo[:num, :]
        next_states = state_combo[num:(2 * num), :]

        cur_prob = mean_prob[:num]
        next_prob = mean_prob[num:]
        prob_gain = next_prob - cur_prob

        # Take out the current time measurements
        batch_size = len(data_dict['num_actions'])

        cur_actions, next_actions = (
            np.zeros((batch_size, self.num_features), dtype=np.uint8)
            for _ in range(2))
        cur_obs, next_obs = (
            np.zeros((batch_size, self.num_features), dtype=np.float32)
            for _ in range(2))

        for idx, (Y, ind_kf, num_action, num_prev_time_action) in enumerate(
                zip(data_dict['Ys'], data_dict['ind_kfs'], data_dict['num_actions'],
                    data_dict['num_prev_time_actions'])):
            if num_action > 0:
                for the_ind_kf, val in zip(ind_kf[-num_action:], Y[-num_action:]):
                    next_obs[idx, the_ind_kf] = val
                    next_actions[idx, the_ind_kf] = 1
            if num_prev_time_action > 0:
                offset = (num_action + num_prev_time_action)
                for the_ind_kf, val in zip(ind_kf[-offset:-num_action], Y[-offset:-num_action]):
                    cur_obs[idx, the_ind_kf] = val
                    cur_actions[idx, the_ind_kf] = 1

        return dict(cur_states=cur_states, next_states=next_states,
                    cur_actions=cur_actions, next_actions=next_actions,
                    cur_obs=cur_obs, next_obs=next_obs, prob_gain=prob_gain, cur_prob=cur_prob,
                    next_prob=next_prob)

    def _construct_cache_feature(self, i, r_input):
        # Save some space. But can not be less than 32 for some reasons...
        cur_state = r_input['cur_states'][i].astype(np.float32)
        next_state = r_input['next_states'][i].astype(np.float32)
        prob_gain = r_input['prob_gain'][i].astype(np.float32)
        cur_prob = r_input['cur_prob'][i].astype(np.float32)
        next_prob = r_input['next_prob'][i].astype(np.float32)

        feature = {
            'cur_state': self._bytes_feature(cur_state),
            'next_state': self._bytes_feature(next_state),
            'prob_gain': self._float_feature(prob_gain),
            'cur_prob': self._float_feature(cur_prob),
            'next_prob': self._float_feature(next_prob),
            'cur_actions': self._bytes_feature(r_input['cur_actions'][i]),
            'next_actions': self._bytes_feature(r_input['next_actions'][i]),
            'cur_obs': self._bytes_feature(r_input['cur_obs'][i]),
            'next_obs': self._bytes_feature(r_input['next_obs'][i]),
            'labels': self._int64_feature(r_input['labels'][i]),
            'patient_inds': self._int64_feature(r_input['patient_inds'][i]),
            'the_steps': self._int64_feature(r_input['the_steps'][i]),
            'total_steps': self._int64_feature(r_input['total_steps'][i]),
            'mortality': self._int64_feature(r_input['mortality'][i]),
        }

        return feature

    def _get_idxes(self, cache_type, cache_proportion, mode):
        ''' Make sure when we cache, we cache all the data we have '''
        return super(MIMIC_per_time_env_exp, self)._get_idxes(
            cache_type='all', cache_proportion=1., mode=mode
        )

    def _cache_experience(self, *args, **kwargs):
        assert 'name' in kwargs
        kwargs['name'] += '_per_time_env'
        return super(MIMIC_per_time_env_exp, self)._cache_experience(*args, **kwargs)


@MGP_RNN_to_RNN_decorator
class MIMIC_per_time_env_exp_rnn(MIMIC_per_time_env_exp):
    pass


class MIMIC_per_time_env_exp_last_obs_rnn(LastObsMixin, MIMIC_per_time_env_exp_rnn):
    pass


if __name__ == '__main__':
    import tensorflow as tf
    import argparse


    def run_MIMIC_discretized_joint_exp():
        parser = argparse.ArgumentParser(description='MIMIC Discreized Exper generaetion')
        parser.add_argument('--identifier', type=str,
                            default='0312-30mins-24hrs-20order-rnn-neg_sampled-with-obs')  # Todo: change Identifier before running
        arr = [ManyToOneRNN, ManyToOneMGP_RNN]
        parser.add_argument('--mgp_rnn_cls', type=str, default='ManyToOneRNN', help=str(arr))
        parser.add_argument('--mgp_rnn_dir', type=str,
                            default='../models/0117-24hours_39feats_38cov_negsampled_rnn-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred24-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN/')
        parser.add_argument('--RL_interval', type=float, default=0.5,
                            help='interval length by hours')
        parser.add_argument('--MIMIC_exp_cls', type=str,
                            default='MIMIC_per_time_env_exp_last_obs_rnn',
                            help='[MIMIC_discretized_joint_exp_random_order, '
                                 'MIMIC_discretized_joint_exp_independent_measurement]')
        parser.add_argument('--MIMIC_cache_exp_cls', type=str,
                            default='MIMIC_cache_discretized_joint_exp_random_order_with_obs',
                            help='[MIMIC_cache_discretized_joint_exp_independent_measurement, '
                                 'MIMIC_cache_discretized_joint_exp_random_order]')
        parser.add_argument('--mode', type=str, default='testing', help='train|test|val|testing|test_onepass_all')
        parser.add_argument('--database_name', type=str, default='mingjie_39features_38covs')
        parser.add_argument('--include_before_death', type=int, default=24)
        parser.add_argument('--num_hours_pred', type=float, default=12.01, help='Num hours pred')
        parser.add_argument('--min_hours_of_patient', type=int, default=12, help='Avoid diff len of exps')
        parser.add_argument('--batch_size', type=int, default=2000)
        parser.add_argument('--cache_type', type=str, default='neg_sampled',
                            help="Choose from ['all', 'pos', 'neg_sampled']")
        parser.add_argument('--cache_proportion', type=float, default=1.)
        parser.add_argument('--num_random_order', type=int, default=20)
        parser.add_argument('--num_pat_produced', type=int, default=30)

        args = parser.parse_args()

        if args.mode == 'test_onepass_all':
            args.num_random_order = -1
            args.cache_proportion = 1.
            args.cache_type = 'all'

        np.random.seed(0)
        tf.set_random_seed(0)

        RL_exp_dir = '../RL_exp_cache/'

        hyperparams = dict(
            reward_func=None, RL_interval=args.RL_interval,
            data_dir='../data/my-mortality/',
            database_name=args.database_name,
            before_end=0.,
            num_hours_warmup=3, min_measurements_in_warmup=5,
            num_hours_pred=args.num_hours_pred, num_X_pred=25, val_ratio=0.15,
            num_gp_samples_limit=300,
            include_before_death=args.include_before_death,
            min_hours_of_patient=args.min_hours_of_patient, verbose=True,
            num_pat_produced=args.num_pat_produced, num_random_order=args.num_random_order,
            mgp_rnn_dir=args.mgp_rnn_dir, mgp_rnn_cls=args.mgp_rnn_cls)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            the_mgp_rnn_class = eval(args.mgp_rnn_cls)
            mgp_rnn = the_mgp_rnn_class.load_saved_mgp_rnn(sess, args.mgp_rnn_dir)

            print('Load model done!')

            mimic_exp_obj = eval(args.MIMIC_exp_cls)
            mimic_exp = mimic_exp_obj(mgp_rnn=mgp_rnn, **hyperparams)

            mimic_exp.cache_experience(sess, args.identifier,
                                       batch_size=args.batch_size,
                                       RL_exp_dir=RL_exp_dir,
                                       mode=args.mode,
                                       cache_proportion=args.cache_proportion,
                                       cache_type=args.cache_type)

            json.dump(hyperparams,
                      open(os.path.join(RL_exp_dir, args.identifier, '%s_hyperparams.json' % args.mode), 'w'))

            if args.mode == 'train' and 'per_time' not in args.MIMIC_exp_cls:
                print('Finish caching. Start normalizing!')

                cache_cls = eval(args.MIMIC_cache_exp_cls)
                cache_dir = os.path.join(RL_exp_dir, args.identifier)
                mimic_exp = cache_cls(cache_dir=cache_dir)

                train_loader = mimic_exp.gen_train_experience(sess, batch_size=1e+7, shuffle=False)
                train_dict = next(iter(train_loader))

                result = {}
                combined = train_dict['cur_state']
                result['mean'] = combined.mean(axis=0)
                result['std'] = combined.std(axis=0)

                pd.DataFrame(result).to_csv(os.path.join(cache_dir, 'state_mean_and_std.csv'), index=None)


    run_MIMIC_discretized_joint_exp()
