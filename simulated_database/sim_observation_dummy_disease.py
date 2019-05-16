'''
Simulate a patient database
'''

import argparse
import datetime
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulated_patient_database import DummyDisease, SimulatedPatientDatabaseDiscrete, DummyClassifier


class ExperienceGeneratorCE:
    def __init__(self, patient_database: 'SimulatedPatientDatabaseDiscrete', classifier: 'DummyClassifier',
                 early_detect_time, split_next_state):
        self.patient_database = patient_database
        self.classifier = classifier
        self.early_detect_time = early_detect_time
        self.split_next_state = split_next_state
        self.action_dim = patient_database.obs_type.num_features
        self.obs_t_dim = patient_database.obs_type.num_datapoints_per_period

    def _split_exp(self, obs, label):
        """
        If make new measure in obs, will be splited in to a new exp without other measurement
        Information gain is calculated as decrease in CE loss due to making new measurement
        """
        split_exps = []

        obs_null = np.copy(obs)
        obs_null[:, -self.action_dim:] = 0  # assume missing value is 0
        ce_loss_null = self.classifier.calculate_ce_loss(obs=obs_null, label=label)

        for i in range(self.action_dim):
            obs_i = np.copy(obs)
            cur_action_i = np.zeros((1, self.action_dim))
            cur_action_i[:, i] = 1
            obs_i[:, -self.action_dim:] = obs_i[:, -self.action_dim:] * cur_action_i  # assume missing value is 0

            if np.any(obs_i[:, -self.action_dim:]):  # skip empty action
                ce_loss_i = self.classifier.calculate_ce_loss(obs=obs_i, label=label)
                split_exps.append([obs_i, cur_action_i, ce_loss_null - ce_loss_i])

        return split_exps

    def _calculate_information_gain(self, obs, label):
        """ Calculate per action information gain compare to not doing any action """
        n = len(obs)
        information_gain_per_action = np.zeros((n, self.action_dim))

        obs_null = np.copy(obs)
        obs_null[:, -self.action_dim:] = self.classifier.missing_value
        ce_loss_null = self.classifier.calculate_ce_loss(obs=obs_null, label=label)

        for i in range(self.action_dim):
            obs_i = np.copy(obs)
            for j in range(self.action_dim):
                if i != j:
                    obs_i[:, - j - 1] = self.classifier.missing_value
            ce_loss_i = self.classifier.calculate_ce_loss(obs=obs_i, label=label)

            information_gain_per_action[:, - i - 1] = (ce_loss_null - ce_loss_i)[:, 0]

        return information_gain_per_action

    def _generate_exp(self, inds):
        data_dict = {k: [] for k in ['cur_state', 'next_state', 'actions', 'labels', 'gain',
                                     'class1_prob', 'ind', 'idx']}
        evaluate_dict = [[], []]

        for i in inds:
            response_progression, measurement, final_label = self.patient_database.all_response_progression[i], \
                                                             self.patient_database.all_measurements[i], \
                                                             self.patient_database.all_labels[i]
            traj_len = len(response_progression)

            get_label = lambda idx: np.array(
                [self.patient_database.all_labels[i] == 1 and idx >= (
                        traj_len - self.obs_t_dim - self.early_detect_time)])[None, :]

            for j in range(traj_len - self.obs_t_dim):
                cur_state = measurement[j: j + self.obs_t_dim, :].reshape(1, -1)
                next_state_joint = measurement[j + 1: j + self.obs_t_dim + 1, :].reshape(1, -1)
                label = get_label(j)
                data_dict['ind'].append(np.array([i]))
                data_dict['idx'].append(np.array([j]))

                if self.split_next_state:
                    split_exps = self._split_exp(next_state_joint, label)
                    evaluate_dict[0].append(next_state_joint)
                    evaluate_dict[1].append(label)

                    for next_state, cur_action, information_gain in split_exps:
                        data_dict['cur_state'].append(cur_state)
                        data_dict['next_state'].append(next_state)
                        data_dict['actions'].append(cur_action)
                        data_dict['labels'].append(label)
                        data_dict['gain'].append(information_gain)
                        # todo: add class1_prob for split_next_state=True
                else:
                    cur_action = (next_state_joint[:, -self.action_dim:] != 0).astype(np.int32)
                    data_dict['cur_state'].append(cur_state)
                    data_dict['next_state'].append(next_state_joint)
                    data_dict['actions'].append(cur_action)
                    data_dict['labels'].append(label)


        data_dict = {k: np.concatenate(data_dict[k], axis=0) for k in data_dict if len(data_dict[k])}

        if self.split_next_state:
            evaluate_dict = [np.concatenate(x, axis=0) for x in evaluate_dict]

        else:
            data_dict['gain'] = self._calculate_information_gain(obs=data_dict['next_state'],
                                                                 label=data_dict['labels'])
            data_dict['class1_prob'] = self.classifier.get_class1_prob(obs=data_dict['next_state'])
            evaluate_dict = [data_dict['next_state'], data_dict['labels']]

        return data_dict, evaluate_dict

    def generate_experience(self, fname=None, val_frac=0.3, save_to_local=False):
        tr_inds, val_inds = self.patient_database._split_dataset(val_frac=val_frac)
        tr_data_dict, evaluate_dict = self._generate_exp(tr_inds)
        val_data_dict, _ = self._generate_exp(val_inds)

        results = {'train': tr_data_dict, 'val': val_data_dict}

        if save_to_local:
            pickle.dump(results, open(fname, "wb"))

        return results, evaluate_dict


class ExperienceGeneratorLinear(ExperienceGeneratorCE):
    """ Change the scheme of information gain culation from CE loss to label weighted probability"""

    def __init__(self, positive_only, **kwargs):
        ExperienceGeneratorCE.__init__(self, **kwargs)

        self.positive_only = positive_only  # remove decrease in prediction probability

    def _split_exp(self, obs, label):
        """
        If make new measure in obs, will be splited in to a new exp without other measurement
        Information gain is calculated as decrease in CE loss due to making new measurement
        """
        split_exps = []

        obs_null = np.copy(obs)
        obs_null[:, -self.action_dim:] = 0  # assume missing value is 0
        prob_null = classifier.get_class1_prob(obs=obs_null)

        for i in range(self.action_dim):
            obs_i = np.copy(obs)
            cur_action_i = np.zeros((1, self.action_dim))
            cur_action_i[:, i] = 1
            obs_i[:, -self.action_dim:] = obs_i[:, -self.action_dim:] * cur_action_i  # assume missing value is 0

            if np.any(obs_i[:, -self.action_dim:]):  # skip empty action
                prob_i = classifier.get_class1_prob(obs=obs_i)  # calculate_ce_loss(obs=obs_i, label=label)
                class_1_gain = (prob_i - prob_null) * label
                class_0_gain = (prob_null - prob_i) * (1 - label)
                split_exps.append([obs_i, cur_action_i, class_1_gain + class_0_gain])

                # todo: support self.positive  here

        return split_exps

    def _calculate_information_gain(self, obs, label):
        """ Calculate per action information gain compare to not doing any action """
        n = len(obs)
        information_gain_per_action = np.zeros((n, self.action_dim))

        obs_null = np.copy(obs)
        obs_null[:, -self.action_dim:] = self.classifier.missing_value
        prob_null = self.classifier.get_class1_prob(obs=obs_null)

        for i in range(self.action_dim):
            obs_i = np.copy(obs)
            for j in range(self.action_dim):
                if i != j:
                    obs_i[:, - j - 1] = self.classifier.missing_value
            prob_i = self.classifier.get_class1_prob(obs=obs_i)
            class_1_gain = (prob_i - prob_null) * label[:, 0]
            class_0_gain = (prob_i - prob_null) * (1 - label)[:, 0]

            if self.positive_only:
                class_1_gain[class_1_gain < 0] = 0
                class_0_gain[class_0_gain < 0] = 0
            else:
                class_0_gain = - class_0_gain

            information_gain_per_action[:, - i - 1] = (class_1_gain + class_0_gain)

        return information_gain_per_action


class ExperienceGeneratorEarlyDetectReward(ExperienceGeneratorLinear):
    def __init__(self, precision_threshold, early_detect_decay, **kwargs):
        ExperienceGeneratorLinear.__init__(self, **kwargs)

        self.precision_threshold = precision_threshold
        self.early_detect_decay = early_detect_decay

    def _get_early_detection_prob_threshold(self, data_dict):
        return self.classifier.find_prob_threshold(
            obs=data_dict['next_state'], label=data_dict['labels'], precision=self.precision_threshold)

    def _add_early_detection_reward(self, data_dict, threshold):
        to_1d = lambda x: x[:, 0] if len(x.shape) == 2 else x
        df = pd.DataFrame({k: to_1d(data_dict[k]) for k in ['ind', 'idx', 'class1_prob', 'labels']})

        traj_len = df.groupby('ind')['idx'].apply(lambda x: np.max(x.values))
        df = df.join(traj_len, on='ind', rsuffix='_max')

        df['detect'] = df.apply(lambda x: x['labels'] == 1 and (x['class1_prob'] >= threshold), axis=1)

        detect_first_time = df.groupby('ind')['detect'].apply(
            lambda x: np.argmax(x.values) if np.max(x.values) == 1 else len(x.values))
        df = df.join(detect_first_time, on='ind', rsuffix='_first_time')

        df['detect_first'] = df['detect_first_time'] == df['idx']

        df['early_detect_reward'] = df.apply(lambda x: (x['idx_max'] + 1 - x['detect_first_time']) *
                                                       (x['detect_first_time'] - x['idx']) ** 0.99
        if x['idx'] < x['detect_first_time'] else 0, axis=1)
        df.groupby('ind')['early_detect_reward'].cumsum(ascending=False)

        data_dict['early_detect_reward'] = df['early_detect_reward'].values

    def generate_experience(self, fname=None, val_frac=0.3, save_to_local=False):
        tr_inds, val_inds = self.patient_database._split_dataset(val_frac=val_frac)
        tr_data_dict, evaluate_dict = self._generate_exp(tr_inds)
        val_data_dict, _ = self._generate_exp(val_inds)

        prob_threshold = self._get_early_detection_prob_threshold(val_data_dict)
        self._add_early_detection_reward(tr_data_dict, prob_threshold)
        self._add_early_detection_reward(val_data_dict, prob_threshold)

        results = {'train': tr_data_dict, 'val': val_data_dict, 'prob_threshold': prob_threshold}

        if save_to_local:
            print('Saving to: {}'.format(fname))
            pickle.dump(results, open(fname, "wb"))

        return results, evaluate_dict


class ExperienceGeneratorProbDiffernetTime:

    def __init__(self, patient_database: 'SimulatedPatientDatabaseDiscrete', classifier: 'DummyClassifier',
                 positive_only, early_detect_time):
        self.patient_database = patient_database
        self.classifier = classifier
        self.early_detect_time = early_detect_time
        self.action_dim = patient_database.obs_type.num_features
        self.obs_t_dim = patient_database.obs_type.num_datapoints_per_period
        self.positive_only = positive_only

    def _calculate_information_gain(self, cur_state, next_state, next_label):
        """ Calculate per action information gain compare to not doing any action """
        n = len(cur_state)
        information_gain_per_action = np.zeros((n, self.action_dim))
        prob_prev = self.classifier.get_class1_prob(obs=cur_state)

        for i in range(self.action_dim):
            obs_i = np.copy(next_state)
            obs_i[:, -self.action_dim:] = cur_state[:, -self.action_dim:]
            obs_i[:, - i - 1] = next_state[:, -i - 1]

            prob_i = self.classifier.get_class1_prob(obs=obs_i)
            class_1_gain = (prob_i - prob_prev) * next_label[:, 0]
            class_0_gain = (prob_i - prob_prev) * (1 - next_label)[:, 0]

            if self.positive_only:
                class_1_gain[class_1_gain < 0] = 0
                class_0_gain[class_0_gain < 0] = 0
            else:
                class_0_gain = - class_0_gain

            information_gain_per_action[:, - i - 1] = (class_1_gain + class_0_gain)

        return information_gain_per_action

    def _generate_exp(self, inds):
        data_dict = {k: [] for k in ['cur_state', 'next_state', 'actions', 'labels', 'gain',
                                     'class1_prob', 'ind', 'idx']}

        for i in inds:
            response_progression, measurement, final_label = self.patient_database.all_response_progression[i], \
                                                             self.patient_database.all_measurements[i], \
                                                             self.patient_database.all_labels[i]
            traj_len = len(response_progression)

            get_label = lambda idx: np.array(
                [self.patient_database.all_labels[i] == 1 and idx >= (
                        traj_len - self.obs_t_dim - self.early_detect_time)])[None, :]

            for j in range(traj_len - self.obs_t_dim):
                cur_state = measurement[j: j + self.obs_t_dim, :].reshape(1, -1)
                next_state_joint = measurement[j + 1: j + self.obs_t_dim + 1, :].reshape(1, -1)
                label = get_label(j)
                data_dict['ind'].append(np.array([i]))
                data_dict['idx'].append(np.array([j]))


                cur_action = (next_state_joint[:, -self.action_dim:] != 0).astype(np.int32)
                data_dict['cur_state'].append(cur_state)
                data_dict['next_state'].append(next_state_joint)
                data_dict['actions'].append(cur_action)
                data_dict['labels'].append(label)

        data_dict = {k: np.concatenate(data_dict[k], axis=0) for k in data_dict if len(data_dict[k])}

        data_dict['gain'] = self._calculate_information_gain(cur_state=data_dict['cur_state'],
                                                             next_state=data_dict['next_state'],
                                                             next_label=data_dict['labels'])

        data_dict['class1_prob'] = self.classifier.get_class1_prob(obs=data_dict['next_state'])

        evaluate_dict = [data_dict['next_state'], data_dict['labels']]

        return data_dict, evaluate_dict

    def generate_experience(self, fname=None, val_frac=0.3, save_to_local=False):
        tr_inds, val_inds = self.patient_database._split_dataset(val_frac=val_frac)
        tr_data_dict, evaluate_dict = self._generate_exp(tr_inds)
        val_data_dict, _ = self._generate_exp(val_inds)

        results = {'train': tr_data_dict, 'val': val_data_dict}

        if save_to_local:
            pickle.dump(results, open(fname, "wb"))

        return results, evaluate_dict


class ExperienceGeneratorMixed:
    def __init__(self, patient_database: 'SimulatedPatientDatabaseDiscrete', classifier: 'DummyClassifier',
                 early_detect_time):
        self.patient_database = patient_database
        self.classifier = classifier
        self.early_detect_time = early_detect_time

        self.action_dim = patient_database.obs_type.num_features
        self.obs_t_dim = patient_database.obs_type.num_datapoints_per_period

    def _calculate_information_gain(self, cur_state, next_state):
        """ Calculate per action information gain compare to not doing any action """

        n = len(cur_state)
        information_gain_per_action = np.zeros((n, self.action_dim))

        prob_cur = self.classifier.get_class1_prob(obs=cur_state)
        prob_next = self.classifier.get_class1_prob(obs=next_state)
        information_gain_true = (prob_next - prob_cur).reshape(-1, 1)

        next_state_null = np.copy(next_state)
        next_state_null[:, -self.action_dim:] = self.classifier.missing_value
        prob_next_null = self.classifier.get_class1_prob(next_state_null)

        for i in range(self.action_dim):
            next_state_i = np.copy(next_state)
            next_state_i[:, -self.action_dim:] = self.classifier.missing_value
            next_state_i[:, -i - 1] = next_state[:, -i - 1]

            prob_next_i = self.classifier.get_class1_prob(obs=next_state_i)
            information_gain_per_action[:, -i - 1] = prob_next_i - prob_next_null

        information_gain_sum = np.sum(information_gain_per_action, axis=1, keepdims=True)
        ratio = information_gain_true / information_gain_sum
        ratio[information_gain_sum == 0] = 0
        information_gain_per_action = information_gain_per_action * ratio
        return information_gain_per_action

    def _generate_exp(self, inds):
        data_dict = {k: [] for k in ['cur_state', 'next_state', 'actions', 'labels', 'gain',
                                     'ind', 'idx']}

        for i in inds:
            response_progression, measurement, final_label = self.patient_database.all_response_progression[i], \
                                                             self.patient_database.all_measurements[i], \
                                                             self.patient_database.all_labels[i]
            traj_len = len(response_progression)

            get_label = lambda idx: np.array(
                [self.patient_database.all_labels[i] == 1 and idx >= (
                        traj_len - self.obs_t_dim - self.early_detect_time)])[None, :]

            for j in range(traj_len - self.obs_t_dim):
                cur_state = measurement[j: j + self.obs_t_dim, :].reshape(1, -1)
                next_state_joint = measurement[j + 1: j + self.obs_t_dim + 1, :].reshape(1, -1)
                label = get_label(j)
                data_dict['ind'].append(np.array([i]))
                data_dict['idx'].append(np.array([j]))

                cur_action = (next_state_joint[:, -self.action_dim:] != 0).astype(np.int32)
                data_dict['cur_state'].append(cur_state)
                data_dict['next_state'].append(next_state_joint)
                data_dict['actions'].append(cur_action)
                data_dict['labels'].append(label)

        data_dict = {k: np.concatenate(data_dict[k], axis=0) for k in data_dict if len(data_dict[k])}

        data_dict['gain'] = self._calculate_information_gain(cur_state=data_dict['cur_state'],
                                                             next_state=data_dict['next_state'])

        # data_dict['class1_prob'] = self.classifier.get_class1_prob(obs=data_dict['next_state'])

        evaluate_dict = [data_dict['next_state'], data_dict['labels']]

        return data_dict, evaluate_dict

    def generate_experience(self, fname=None, val_frac=0.3, save_to_local=False):
        tr_inds, val_inds = self.patient_database._split_dataset(val_frac=val_frac)
        tr_data_dict, evaluate_dict = self._generate_exp(tr_inds)
        val_data_dict, _ = self._generate_exp(val_inds)

        results = {'train': tr_data_dict, 'val': val_data_dict}

        if save_to_local:
            pickle.dump(results, open(fname, "wb"))

        return results, evaluate_dict

    def evaluate(self, get_action, reward_func, threshold=0.5):
        acc_rewards = np.zeros((self.patient_database.num_patients, self.action_dim))
        acc_actions = np.zeros((self.patient_database.num_patients, self.action_dim))
        accuracy = np.zeros((self.patient_database.num_patients))

        for i in range(self.patient_database.num_patients):
            labels, responses, measurement = self.patient_database.all_labels[i], \
                                             self.patient_database.all_response_progression[i], \
                                             self.patient_database.all_measurements[i]

            traj_len = len(responses)

            actions = np.zeros((traj_len, self.action_dim))
            actions[:self.obs_t_dim, :] = 1

            obs_cur_state = measurement[:self.obs_t_dim, :].reshape(1, -1)

            for j in range(traj_len - self.obs_t_dim):
                cur_action = get_action(obs_cur_state)

                actions[j + self.obs_t_dim, :] = cur_action
                next_mask = actions[j + 1: j + 1 + self.obs_t_dim, :]
                true_next_state = measurement[j + 1: j + 1 + self.obs_t_dim, :]
                obs_next_state = (next_mask * true_next_state).reshape(1, -1)  # assume missing value = 0

                information_gain = self._calculate_information_gain(cur_state=obs_cur_state, next_state=obs_next_state)

                # acc_action += cur_action
                acc_actions[i] += cur_action[0] / (traj_len - self.obs_t_dim)
                acc_rewards[i] += reward_func(cur_action=cur_action, information_gain=information_gain)[0]

                obs_cur_state = obs_next_state

            accuracy[i] = (self.classifier.get_class1_prob(obs_cur_state) > threshold) == labels

            if i % 10 == 0:
                print('finished: {}/{}'.format(i, self.patient_database.num_patients))

        return acc_rewards, acc_actions, accuracy

    def evaluate_batch(self, get_action, reward_func, threshold=0.5):
        traj_len = list(map(len, self.patient_database.all_response_progression))
        max_traj_len = max(traj_len)

        rewards = np.zeros((self.patient_database.num_patients, self.action_dim, max_traj_len))
        actions = np.zeros((self.patient_database.num_patients, self.action_dim, max_traj_len))
        actions[:, :, :self.obs_t_dim] = 1

        obs_cur_state = np.concatenate(list(map(lambda x: x[:self.obs_t_dim, :].reshape(1, -1),
                                                self.patient_database.all_measurements)))

        for j in range(max_traj_len - self.obs_t_dim):
            cur_action = get_action(obs_cur_state)
            actions[:, :, j + self.obs_t_dim] = cur_action
            next_mask = actions[:, :, j + 1: j + 1 + self.obs_t_dim]
            get_next_state = lambda x: x[j + 1: j + 1 + self.obs_t_dim, :] * next_mask
            obs_next_state = map(get_next_state, self.patient_database.all_measurements)
            information_gain = self._calculate_information_gain(cur_state=obs_cur_state, next_state=obs_next_state)
            rewards += reward_func(cur_action=cur_action, information_gain=information_gain)
            obs_cur_state = obs_next_state

            if j % 5 == 0:
                print('finished: {}/{}'.format(j, max_traj_len - self.obs_t_dim))

    @staticmethod
    def evaluate_with_deterministic_statte(get_action):
        measurement = np.ones((20, 6)) * 0
        measurement[:5, :] = -1

        measurement[-5:, :] = 1

        cur_state = np.stack([measurement[i:i + 5, :].reshape(-1) for i in range(20 - 5)], axis=0)
        return cur_state, get_action(cur_state)


class ExperienceGeneratorSequential:
    def __init__(self,
                 patient_database: 'SimulatedPatientDatabaseDiscrete',
                 classifier: 'DummyClassifier',
                 early_detect_time,
                 num_orders,
                 num_end_time_use=-1,
                 redundant_rate=0):

        self.patient_database = patient_database
        self.classifier = classifier
        self.early_detect_time = early_detect_time

        self.action_dim = patient_database.obs_type.num_features
        self.obs_t_dim = patient_database.obs_type.num_datapoints_per_period
        self.redundant_rate = redundant_rate
        self.num_end_time_use = num_end_time_use
        self.num_orders = num_orders

    def unwarp_state(self, state):
        return state[:, :-self.classifier.num_features], state[:, -self.classifier.num_features:]

    def _generate_random_order_splited_exp(self, cur_state, all_action, next_state):
        """
        Return a data dict containing sequential addition of measurement made given joint_cur_state and joint_next_state
        """
        # go to next time point
        cur_history_action = (cur_state[:, -self.action_dim:] != self.classifier.missing_value).astype(int)
        next_history_action = np.zeros((1, self.action_dim))
        action = np.zeros((1, self.action_dim + 1))
        action[:, -1] = 1

        next_state_new = np.copy(next_state)
        next_time_idx = len(next_state_new[0]) - self.action_dim
        next_state_new[:, next_time_idx:] = self.classifier.missing_value

        data_dict = {'cur_state'          : [cur_state],
                     'next_state'         : [np.copy(next_state_new)],
                     'cur_history_action' : [cur_history_action],
                     'next_history_action': [next_history_action],
                     'action'             : [action]}

        # make measurement at the next time point
        history_action = np.zeros((1, self.action_dim))

        measurements = [i for i, made in enumerate(all_action[0]) if made]
        random.shuffle(measurements)

        for m in measurements:
            data_dict['cur_state'].append(np.copy(next_state_new))

            next_state_new[:, next_time_idx + m] = next_state[:, next_time_idx + m]
            data_dict['next_state'].append(np.copy(next_state_new))

            data_dict['cur_history_action'].append(np.copy(history_action))
            history_action[:, m] = 1
            data_dict['next_history_action'].append(np.copy(history_action))

            action = np.zeros((1, self.action_dim + 1))
            action[:, m] = 1
            data_dict['action'].append(action)

        return data_dict

    def _generate_delay_state(self, cur_state, next_state):
        return cur_state[:, :-self.action_dim], next_state[:, :-self.action_dim]

    def _calculate_information_gain(self, cur_state, next_state):
        prob_cur = self.classifier.get_class1_prob(cur_state)
        prob_next = self.classifier.get_class1_prob(next_state)

        return (prob_next - prob_cur)[:, None]

    def _generate_redundant_exp(self, data_dict):
        """ Add artificial exp that representing Do the same action on time point """
        new_data_dict = {k: [] for k in data_dict}
        new_data_dict['cur_state'] = data_dict['next_state'][1:]
        new_data_dict['next_state'] = data_dict['next_state'][1:]
        new_data_dict['cur_history_action'] = data_dict['next_history_action'][1:]
        new_data_dict['next_history_action'] = data_dict['next_history_action'][1:]
        new_data_dict['action'] = data_dict['action'][1:]

        return new_data_dict

    def _generate_exp(self, inds):
        data_dict = {k: [] for k in ['cur_state', 'next_state', 'action',
                                     'cur_history_action', 'next_history_action',
                                     'label', 'gain', 'ind', 'redundant', 'final_label']}
        # data_dict['num_exp'] = 0

        for i in inds:  #
            response_progression, measurement, final_label = self.patient_database.all_response_progression[i], \
                                                             self.patient_database.all_measurements[i], \
                                                             np.array([[self.patient_database.all_labels[i] == 1]])
            traj_len = len(response_progression)
            starting_j = traj_len - self.num_end_time_use if self.num_end_time_use != -1 else 0

            get_label = lambda idx: final_label and np.array([[idx >= (traj_len - self.early_detect_time)]])

            assert final_label == response_progression[-1]

            for j in range(starting_j, traj_len):
                if j < self.obs_t_dim:
                    padding_cur = np.zeros((1, self.action_dim * (self.obs_t_dim - j)))
                    padding_next = np.zeros((1, self.action_dim * (self.obs_t_dim - j - 1)))
                    cur_state = measurement[:j, :].reshape(1, -1)
                    next_state = measurement[: j + 1, :].reshape(1, -1)

                    cur_state = np.concatenate((padding_cur, cur_state), axis=1)
                    next_state = np.concatenate((padding_next, next_state), axis=1)
                else:
                    cur_state = measurement[j - self.obs_t_dim: j, :].reshape(1, -1)
                    next_state = measurement[j - self.obs_t_dim + 1: j + 1, :].reshape(1, -1)
                label = get_label(j + 1)

                cur_action = (next_state[:, -self.action_dim:] != 0).astype(np.int32)

                for p in range(self.num_orders):
                    splited_exp = self._generate_random_order_splited_exp(cur_state=cur_state, all_action=cur_action,
                                                                      next_state=next_state)

                    for k in splited_exp:
                        data_dict[k] += splited_exp[k]

                num_exp = len(splited_exp['cur_state']) * self.num_orders
                data_dict['ind'] += [np.array([[i]])] * num_exp
                data_dict['label'] += [label] * num_exp
                data_dict['redundant'] += [np.zeros((1, 1))] * num_exp
                data_dict['final_label'] += [final_label] * num_exp

        data_dict = {k: np.concatenate(data_dict[k], axis=0) for k in data_dict if len(data_dict[k])}

        data_dict['gain'] = self._calculate_information_gain(cur_state=data_dict['cur_state'],
                                                             next_state=data_dict['next_state'])
        data_dict['delay_cur_state'], data_dict['delay_next_state'] = \
            self._generate_delay_state(data_dict['cur_state'], data_dict['next_state'])

        evaluate_dict = [data_dict['next_state'], data_dict['label']]

        return data_dict, evaluate_dict

    def generate_experience(self, fname=None, val_frac=0.3, save_to_local=False):
        tr_inds, val_inds = self.patient_database._split_dataset(val_frac=val_frac)
        tr_data_dict, evaluate_dict = self._generate_exp(tr_inds)
        val_data_dict, _ = self._generate_exp(val_inds)

        results = {'train': tr_data_dict, 'val': val_data_dict}

        if save_to_local:
            pickle.dump(results, open(fname, "wb"))

        return results, evaluate_dict

    def evaluate_batch(self, get_action, reward_func, n_samples=None):
        n_samples = len(self.patient_database.all_response_progression) if n_samples is None else n_samples
        response_progression = self.patient_database.all_response_progression[:n_samples]
        t_len = np.array(list(map(len, response_progression))).astype(int)
        max_t_len = max(t_len)

        cur_t = np.zeros(n_samples, dtype=int)
        rewards = np.zeros((n_samples, max_t_len))
        actions = np.zeros((n_samples, max_t_len, self.action_dim + 1), dtype=int)
        gain = np.zeros((n_samples, max_t_len, self.action_dim + 1))
        gain_class1 = np.zeros((n_samples, max_t_len, self.action_dim + 1))
        cur_prob = np.zeros((n_samples, max_t_len))

        is_not_finished = lambda actions: actions[np.arange(n_samples), t_len - 1, -1] != 1

        def get_next_state(all_actions, t_idx):
            assert np.all(t_idx < t_len)
            next_state = []
            for i in range(-self.obs_t_dim + 1, 1):
                padding = ((t_idx + i) < 0).reshape(-1, 1)  # pad all time poeint with 4 previous measurement
                next_state_mask = all_actions[np.arange(n_samples), t_idx + i, :-1]
                true_next_state = np.stack([self.patient_database.all_measurements[j][t_idx[j] + i, :]
                                            for j in range(n_samples)])
                next_state.append(next_state_mask * true_next_state * (1 - padding) + np.zeros(next_state_mask.shape))

            next_state = np.stack(next_state, axis=1).reshape(n_samples, -1)
            next_state = np.concatenate([next_state, next_state_mask], axis=1)
            return next_state

        def get_delay_cur_state(raw_cur_state, history_action):
            return np.concatenate((raw_cur_state[:, :-self.action_dim], history_action), axis=1)

        cur_state = get_next_state(all_actions=actions, t_idx=cur_t)

        while any(is_not_finished(actions)):
            unfinished = is_not_finished(actions)
            raw_cur_state, history_action = self.unwarp_state(cur_state)

            delay_cur_state = get_delay_cur_state(raw_cur_state, history_action)
            cur_action = get_action(delay_cur_state=delay_cur_state, history_action=history_action, done=~unfinished)
            actions[unfinished, cur_t[unfinished], :] = \
                np.clip(cur_action[unfinished] + actions[unfinished, cur_t[unfinished], :], 0, 1)

            next_t = np.minimum(cur_action[:, -1] + cur_t, t_len - 1)

            next_state = get_next_state(all_actions=actions, t_idx=next_t)
            raw_next_state, _ = self.unwarp_state(next_state)

            information_gain = self._calculate_information_gain(cur_state=raw_cur_state, next_state=raw_next_state)

            label = np.array([response[t] for t, response in zip(next_t, response_progression)]).reshape(-1, 1)
            gain[unfinished, cur_t[unfinished], :] += information_gain[unfinished] * cur_action[unfinished] * (
                        2 * label[unfinished] - 1)
            gain_class1[unfinished, cur_t[unfinished], :] += information_gain[unfinished] * cur_action[unfinished] * \
                                                             label[unfinished]
            rewards[unfinished, cur_t[unfinished]] += reward_func(cur_action=cur_action[unfinished],
                                                                  information_gain=information_gain[unfinished],
                                                                  label=label[unfinished])[:, 0]
            cur_prob[unfinished, cur_t[unfinished]] = self.classifier.get_class1_prob(obs=raw_cur_state[unfinished])

            cur_state = next_state
            cur_t = next_t

        results = {
            't_len'               : t_len,
            'rewards'             : rewards,
            'actions'             : actions,
            'gain'                : gain,
            'gain_class1'         : gain_class1,
            'cur_prob'            : cur_prob,
            'response_progression': response_progression,
            'dead'                : np.array([np.any(traj) for traj in response_progression])
        }
        return results

    @staticmethod
    def _calculate_accumulative_reward(rewards, gamma):
        gamma_rollout = np.array([gamma ** i for i in range(rewards.shape[1])]).reshape(1, -1)
        cum_reward = np.sum(rewards * gamma_rollout, axis=1)

        print(f'Avg/std reward: {np.mean(cum_reward)} +/- {np.std(cum_reward)}')

    @staticmethod
    def _calculate_action_freq(actions, t_len):
        action_freq_joint_by_time = np.sum(actions[:, :, :-1], axis=1) / t_len[:, None]
        action_freq_per_action = np.mean(action_freq_joint_by_time, axis=0)
        action_freq_all_action = np.mean(action_freq_joint_by_time)

        print(f'Action freq (pre action): {action_freq_per_action}')
        print(f'Action freq (joint): {action_freq_all_action}')

    @staticmethod
    def _summarize_class1_gain(dead, gain_class1):
        avg_gain_class1 = np.mean(np.sum(gain_class1[dead], axis=(1, 2)))
        print(f'Average gain for class 1 among dead patient: {avg_gain_class1}')

    @staticmethod
    def _demo_dead_patient(dead, response_progression, cur_prob, t_len, actions, num_demo=1):
        response_progression = [traj for traj, d in zip(response_progression, dead) if d]
        cur_prob = cur_prob[dead]
        t_len = t_len[dead]
        actions = actions[dead]

        for i in range(num_demo):
            traj = response_progression[i]
            traj_len = t_len[i]
            prob = cur_prob[i][:traj_len]
            t_idx = np.arange(traj_len)
            cur_action = actions[i]

            plt.figure()
            plt.subplot(211)
            plt.plot(t_idx, traj)
            plt.subplot(212)
            plt.plot(t_idx, prob)
            plt.show()

        print('done')

    def summarize(self, dic, save_to_local=False):
        self._calculate_accumulative_reward(rewards=dic['rewards'], gamma=dic['gamma'])
        self._calculate_action_freq(dic['actions'], dic['t_len'])
        self._summarize_class1_gain(dic['dead'], dic['gain_class1'])
        self._demo_dead_patient(dic['dead'], dic['response_progression'],
                                dic['cur_prob'], dic['t_len'], dic['actions'], num_demo=2)

        if save_to_local:
            print('')


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate dummy dataset for dqn training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_num_terminal_states', type=int, default=5)
    parser.add_argument('--early_detect_time', type=int, default=5)
    parser.add_argument('--feature_noise', type=float, default=0.1)

    parser.add_argument('--num_useful_features', type=int, default=5)
    parser.add_argument('--num_noisy_features', type=int, default=5)
    parser.add_argument('--num_datapoints_per_period', type=int, default=5)
    parser.add_argument('--period_length', type=int, default=1)
    parser.add_argument('--min_periods', type=int, default=5)
    parser.add_argument('--max_periods', type=int, default=10)
    parser.add_argument('--num_patients', type=int, default=5000)
    parser.add_argument('--information_decay_rate', type=float, default=0.5)
    parser.add_argument('--redundant_rate', type=int, default=0.0)
    parser.add_argument('--num_end_time_use', type=int, default=5)

    parser.add_argument('--precision_threshold', type=float, default=0.9)
    parser.add_argument('--num_orders', type=int, default=3)
    # parser.add_argument('--precision_threshold', type=float, default=0.99)
    # parser.add_argument('--split_next_state', type=int, default=0)

    # parser.add_argument('--positive_only', type=int, default=0)
    # parser.add_argument('--early_detect_decay', type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    date = datetime.datetime.now().strftime('%m%d')
    dirname = '../data/simulated_disease/'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    pickle.dump(args, open(dirname + date + '_hyperparameters.pkl', 'wb'))

    classifier = DummyClassifier(num_useful_features=args.num_useful_features,
                                 num_noisy_features=args.num_noisy_features,
                                 obs_t_dim=args.max_num_terminal_states,
                                 information_decay_rate=args.information_decay_rate)

    def run_simulation_mixed(keep_rate, not_debug=True):
        disease = DummyDisease(feature_noise=args.feature_noise,
                               max_num_terminal_states=args.max_num_terminal_states,
                               num_useful_features=args.num_useful_features,
                               num_noisy_features=args.num_noisy_features,
                               num_datapoints_per_period=args.num_datapoints_per_period,
                               period_length=args.period_length,
                               min_periods=args.min_periods,
                               max_periods=args.max_periods,
                               keep_rate=keep_rate)

        patient_database = \
            SimulatedPatientDatabaseDiscrete(obs_type=disease, num_patients=args.num_patients)

        exp_generator = \
            ExperienceGeneratorSequential(patient_database=patient_database,
                                          classifier=classifier,
                                          early_detect_time=args.early_detect_time,
                                          redundant_rate=args.redundant_rate,
                                          num_end_time_use=args.num_end_time_use,
                                          num_orders=args.num_orders)

        fname = dirname + date + '_simulated_features_k{}_mixed.pkl'.format(disease.keep_rate)

        data_dict, evaluate_dict = exp_generator.generate_experience(fname=fname, save_to_local=not_debug)

        print('Train death rate {}, test death rate {}'.format(
            np.mean(data_dict['train']['label']), np.mean(data_dict['val']['label'])))

        aupr = classifier.calculate_aupr(obs=evaluate_dict[0], label=evaluate_dict[1])
        print('Keep rate: {}, AUPR: {}'.format(disease.keep_rate, aupr))

        threshold = classifier.find_prob_threshold(obs=evaluate_dict[0], label=evaluate_dict[1],
                                                   precision=args.precision_threshold)

        print('Prediction probability threshold for precision {}: {}'.format(args.precision_threshold, threshold))


    def run_simulation_sequencial(keep_rate, not_debug=True):
        disease = DummyDisease(feature_noise=args.feature_noise,
                               max_num_terminal_states=args.max_num_terminal_states,
                               num_useful_features=args.num_useful_features,
                               num_noisy_features=args.num_noisy_features,
                               num_datapoints_per_period=args.num_datapoints_per_period,
                               period_length=args.period_length,
                               min_periods=args.min_periods,
                               max_periods=args.max_periods,
                               keep_rate=keep_rate)

        patient_database = \
            SimulatedPatientDatabaseDiscrete(obs_type=disease, num_patients=args.num_patients)

        print('Dead rate: {}'.format(np.mean(patient_database.all_labels)))

        exp_generator = \
            ExperienceGeneratorSequential(patient_database=patient_database,
                                          classifier=classifier,
                                          early_detect_time=args.early_detect_time,
                                          redundant_rate=args.redundant_rate,
                                          num_end_time_use=args.num_end_time_use,
                                          num_orders=args.num_orders)

        fname = dirname + date + '_simulated_features_k{}_sequential.pkl'.format(disease.keep_rate)

        data_dict, evaluate_dict = exp_generator.generate_experience(fname=fname, save_to_local=not_debug)

        aupr = classifier.calculate_aupr(obs=evaluate_dict[0], label=evaluate_dict[1])
        print('Keep rate: {}, AUPR: {}'.format(disease.keep_rate, aupr))


    # run_simulation_mixed(keep_rate=0.9)
    run_simulation_sequencial(keep_rate=0.5)

    print("done")
