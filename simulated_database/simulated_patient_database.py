import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from utils.math import ou_kernel_np, sigmoid


def generate_mgp_params(num_features, mean_scale=1.0, noise_scale=1.0, mean_range=(0, 1)):
    """
    Generate kernel function and noise level for a multitask gp
    """

    tmp = np.random.normal(0, 1, (num_features, num_features))
    kfs = np.corrcoef(tmp)

    means = mean_scale * np.linspace(mean_range[0], mean_range[1], num_features)
    noises = np.random.rand(num_features) * noise_scale

    return [means, kfs, noises]


class TimeSeriesObservation:
    def __init__(self, num_useful_features, num_noisy_features, max_periods, min_periods, period_length,
                 num_datapoints_per_period, keep_rate, preset_params=None):

        self.num_useful_features = num_useful_features
        self.num_noisy_features = num_noisy_features
        self.num_features = num_noisy_features + num_useful_features

        self.max_periods = max_periods
        self.min_periods = min_periods
        self.period_length = period_length
        self.num_datapoints_per_period = num_datapoints_per_period
        self.keep_rate = keep_rate

        self.x = np.arange(
            self.period_length * self.max_periods * self.num_datapoints_per_period) / self.period_length / self.num_datapoints_per_period

        if 'num_states' not in self.__dict__:
            self.num_states = None
        self.params = preset_params
        if preset_params is None:
            self._setup_params()

    def _setup_params(self):
        raise NotImplementedError

    def _sim_response_progression(self):
        raise NotImplementedError

    def _sim_measurements(self, response_progression):
        raise NotImplementedError

    def _subset_measurements(self, response_progression, measurements):
        raise NotImplementedError

    def _drop_observation_random(self, measurements):
        raise NotImplementedError

    def create_patient_data(self):
        raise NotImplementedError


class Disease(TimeSeriesObservation):
    """
    Describe the disease states and their stats transition function as well as features that is related to
    the disease states

    max_num_terminal_states: maximum number of terminal state
        0: infinite number of terminal state
        >0: will cut the response progression up to the requires number
    """

    def __init__(self, max_num_terminal_states,
                 transition_prob=np.array([0.9, 0.1, 0,
                                           0.1, 0.8, 0.1,
                                           0.0, 0.0, 1.0]).reshape(3, 3),
                 **kwargs):

        self.num_states = len(transition_prob)
        self.transition_prob = transition_prob
        self.max_num_terminal_states = max_num_terminal_states

        super(Disease, self).__init__(**kwargs)

    def _setup_params(self):
        """ Generate multitask gp parameters for each disease state """

        self.params = [
            generate_mgp_params(self.num_useful_features, mean_scale=5 * i, noise_scale=0.1, mean_range=(0, 1))
            for i in range(self.num_states)]

        self.noisy_params = generate_mgp_params(self.num_noisy_features, mean_scale=5, noise_scale=0.1,
                                                mean_range=(0, 1))

    # todo: allow removal of observation after reaching last state
    def _sim_response_progression(self):
        """ Simulate discrete disease state progression """

        response_progression = np.zeros(self.max_periods, dtype=np.int32)

        for i in range(self.max_periods - 1):
            response_progression[i + 1] = np.random.choice(self.num_states,
                                                           p=self.transition_prob[response_progression[i], :])

        full_response_progression = np.repeat(response_progression, self.num_datapoints_per_period)

        return full_response_progression

    def _sim_measurements(self, response_progression):

        all_measurements = [[], [], []]

        xi = np.arange(self.num_datapoints_per_period) / self.period_length / self.num_datapoints_per_period

        for i in range(self.max_periods):
            stage = response_progression[i * self.num_datapoints_per_period]
            gp_data = sim_multitask_gp(xi + i * self.period_length,
                                       kernel_fn=ou_kernel_np,
                                       length=1.0,
                                       means=self.params[stage][0],
                                       cov_f=self.params[stage][1],
                                       noise_vars=self.params[stage][2])

            gp_data[2] += i * self.num_datapoints_per_period

            noisy_gp_data = sim_multitask_gp(xi + i * self.period_length,
                                             kernel_fn=ou_kernel_np,
                                             length=1.0,
                                             means=self.noisy_params[0],
                                             cov_f=self.noisy_params[1],
                                             noise_vars=self.noisy_params[2])

            noisy_gp_data[1] += self.num_useful_features
            noisy_gp_data[2] += i * self.num_datapoints_per_period

            all_measurements[0] = np.concatenate((all_measurements[0], gp_data[0], noisy_gp_data[0]))
            all_measurements[1] = np.concatenate((all_measurements[1], gp_data[1], noisy_gp_data[1]))
            all_measurements[2] = np.concatenate((all_measurements[2], gp_data[2], noisy_gp_data[2]))

        return sort_sparse_array(all_measurements[0],
                                 all_measurements[1].astype(np.int32), all_measurements[2].astype(np.int32))

    def _subset_measurements(self, response_progression, measurements):
        """ Subset patient's trajectory in random length """

        num_subset_periods = np.random.rand() * (self.max_periods - self.min_periods) + self.min_periods
        num_t = int(num_subset_periods * self.num_datapoints_per_period)

        response_progression = response_progression[:num_t]

        measurements = [item[:num_t * self.num_features] for item in measurements]

        return response_progression, measurements

    def _drop_observation_random(self, measurements):
        """ Randomly drop some fraction of fully observed time series for a patient's time series data """

        perm = np.random.permutation(len(measurements[0]))
        n_train = int(self.keep_rate * len(measurements[0]))
        train_inds = np.sort(perm[:n_train])

        return [item[train_inds] for item in measurements]

    def _cut_multiple_terminal_states(self, response_progression, measurements):
        label = 0
        terminal_states = [idx for idx, item in enumerate(response_progression) if item == self.num_states - 1]
        if len(terminal_states) > self.max_num_terminal_states:
            response_progression = response_progression[:terminal_states[0] + self.max_num_terminal_states]

            subset_idx = measurements[2] <= terminal_states[self.max_num_terminal_states - 1]
            measurements = [item[subset_idx] for item in measurements]
            label = 1

        return label, response_progression, measurements

    def create_patient_data(self):
        response_progression = \
            self._sim_response_progression()

        measurements = \
            self._sim_measurements(response_progression=response_progression)

        response_progression, measurements = \
            self._subset_measurements(response_progression=response_progression, measurements=measurements)

        measurements = \
            self._drop_observation_random(measurements=measurements)

        label = None

        if self.max_num_terminal_states > 0:
            label, response_progression, measurements = \
                self._cut_multiple_terminal_states(response_progression=response_progression, measurements=measurements)

        return label, response_progression, measurements


class DummyDisease(Disease):
    """
    Masurement:
    -1: no problem
    0: unmeasured
    1: problem
    """

    def __init__(self, feature_noise, **kwargs):
        self.feature_noise = feature_noise
        self.missing_value = 0
        super(DummyDisease, self).__init__(transition_prob=np.array([0.98, 0.02, 0.0, 1.0]).reshape(2, 2), **kwargs)

    def _setup_params(self):
        self.params = lambda state: (np.ones(self.num_useful_features) if state == self.num_states - 1
                                     else np.ones(self.num_useful_features) * -1) + \
                                    np.random.rand(self.num_useful_features) * self.feature_noise

        self.noisy_params = lambda: np.random.choice([-1, 1], self.num_noisy_features) + \
                                    np.random.rand(self.num_noisy_features) * self.feature_noise

    def _sim_measurements(self, response_progression):
        measurements = np.zeros((self.num_datapoints_per_period * self.max_periods, self.num_features))

        # measurements[:, -self.num_noisy_features:] = np.stack([fun() for fun in self.noisy_params], axis=-1)

        for i in range(self.max_periods):
            state = response_progression[i * self.num_datapoints_per_period]
            start_idx = i * self.num_datapoints_per_period

            for j in range(self.num_datapoints_per_period):
                measurements[start_idx + j, :self.num_useful_features] = self.params(state=state)
                measurements[start_idx + j, -self.num_noisy_features:] = self.noisy_params()

        return measurements

    def _subset_measurements(self, response_progression, measurements):
        num_subset_periods = np.random.rand() * (self.max_periods - self.min_periods) + self.min_periods
        num_t = int(num_subset_periods) * self.num_datapoints_per_period

        response_progression = response_progression[:num_t]
        measurements = measurements[:num_t, :]

        return response_progression, measurements

    def _drop_observation_random(self, measurements):
        measurements = measurements.reshape(-1)
        perm = np.random.permutation(len(measurements))
        n_drop = int((1 - self.keep_rate) * len(measurements))
        if n_drop > 0:
            drop_inds = np.sort(perm[-n_drop:])
            measurements[drop_inds] = self.missing_value
        return measurements.reshape(-1, self.num_features)

    def _cut_multiple_terminal_states(self, response_progression, measurements):
        label = 0
        terminal_states = [idx for idx, item in enumerate(response_progression) if item == self.num_states - 1]

        if len(terminal_states) >= self.max_num_terminal_states:
            label = 1
            response_progression = response_progression[:terminal_states[0] + self.max_num_terminal_states]
            measurements = measurements[:(terminal_states[0] + self.max_num_terminal_states), :]

        return label, response_progression, measurements


class Classifier():
    def __init__(self, num_useful_features, num_noisy_features, obs_t_dim):
        self.num_useful_features = num_useful_features
        self.num_noisy_features = num_noisy_features
        self.num_features = self.num_noisy_features + self.num_useful_features
        self.obs_t_dim = obs_t_dim
        self.missing_value = 0
        self._set_up_classier()

    def _set_up_classier(self):
        raise NotImplementedError

    def get_prediction_prob(self, obs, label):
        raise NotImplementedError

class DummyClassifier(Classifier):

    def __init__(self, information_decay_rate, **kwargs):
        self.information_decay_rate = information_decay_rate

        Classifier.__init__(self, **kwargs)

    def _set_up_classier(self):
        self.feature_weights = self._get_decay_feature_weights()
        self.max_score = self._get_max_score()

    def _get_feature_weights(self):
        useful_feature_weights = np.arange(self.num_useful_features) + 1
        noisy_feature_weights = np.zeros(self.num_noisy_features)

        return np.tile(np.concatenate((useful_feature_weights, noisy_feature_weights)), self.obs_t_dim)

    def _get_decay_feature_weights(self):
        useful_feature_weights = np.arange(self.num_useful_features) + 1
        noisy_feature_weights = np.zeros(self.num_noisy_features)

        feature_weights_per_t = np.concatenate([useful_feature_weights, noisy_feature_weights])
        return np.concatenate([feature_weights_per_t * self.information_decay_rate ** (self.obs_t_dim - i - 1)
                               for i in range(self.obs_t_dim)])

    def _get_score(self, obs):
        return np.sum(obs * self.feature_weights, axis=1)

    def _get_max_score(self):
        optimal_obs = np.ones((1, self.num_features * self.obs_t_dim))
        return self._get_score(optimal_obs)

    def get_class1_prob(self, obs):
        return sigmoid(self._get_score(obs) / self.max_score)

    def get_prediction_prob(self, obs, label=None):
        class_1_prob = self.get_class1_prob(obs=obs)
        return np.stack((1 - class_1_prob, class_1_prob), axis=-1)

    def calculate_ce_loss(self, obs, label, epsilon=1e-12):
        pred_prob = self.get_prediction_prob(obs, label)
        pred_prob = np.clip(pred_prob, epsilon, 1. - epsilon)
        return - np.log(pred_prob[:, 0:1]) * (1 - label) - label * np.log(pred_prob[:, 1:2])

    def calculate_aupr(self, obs, label):
        pred_prob = self.get_prediction_prob(obs)
        return average_precision_score(y_true=np.concatenate((1 - label, label), axis=1), y_score=pred_prob)

    def find_prob_threshold(self, obs, label, precision):
        """ Return the probability threshold for the precision picked under a AUPR curve """
        pred_prob = self.get_prediction_prob(obs)
        precisions, recalls, thresholds = precision_recall_curve(y_true=label, probas_pred=pred_prob[:, 1])

        idx = np.argmax(precisions >= precision)

        threshold_picked, precision_picked, recall_picked = thresholds[idx], precisions[idx], recalls[idx]

        print('Picked - precision: {}, recall: {}, threshold: {}'.format(
            precision_picked, recall_picked, threshold_picked))

        return threshold_picked


class DummyClassifierRandomForest(Classifier):
    """ Problem with this classifier, need to modified observation to set up class specific feature importance """

    def __init__(self, **kwargs):
        Classifier.__init__(self, **kwargs)

    def _set_up_classier(self):
        self.trained = False
        self.clf = RandomForestClassifier(n_estimators=1, random_state=0)

    def _train(self, obs, label):
        self.trained = True
        self.clf.fit(obs, label[:, 0])

    def get_prediction_prob(self, obs, label):
        if not self.trained:
            self._train(obs, label)

        return self.clf.predict_proba(obs)


class Glucose(TimeSeriesObservation):
    """ Describe the blood glucose level and its relationship with some features """
    def __init__(self, **kwargs):
        super(Glucose, self).__init__(**kwargs)


def sort_sparse_array(y, ind_kf, ind_kt):
    sorted_order = np.lexsort((ind_kf, ind_kt))
    return np.take(y, sorted_order), np.take(ind_kf, sorted_order), np.take(ind_kt, sorted_order)


def sim_multitask_gp(times, kernel_fn, length, means, cov_f, noise_vars):
    """
    Draw samples from a multitask gp.
    """

    num_features = np.shape(cov_f)[0]
    num_observed_time = len(times)          # Num of observed time points
    n = num_observed_time * num_features    # Number of data points

    # given covariance of time and covariance of each variable, produce the covariance of the Mgp
    cov_t = kernel_fn(times, length)
    sigma = np.diag(noise_vars)

    cov = np.kron(cov_f, cov_t) + np.kron(sigma, np.eye(num_observed_time)) + 1e-6 * np.eye(n)
    l_cov = np.linalg.cholesky(cov)

    y = np.dot(l_cov, np.random.normal(0, 1, n))  # Draw samples from N(0, 1)

    # get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(num_features), (num_observed_time, 1)).flatten('F')  # vec by column
    ind_kt = np.tile(np.arange(num_observed_time), (num_features, 1)).flatten()

    # sort by time then by output
    # y, ind_kf, ind_kt = sort_sparse_array(y, ind_kf, ind_kt)

    # Add mean to y
    y = np.array(y) + np.tile(means, num_observed_time)

    return [y, np.array(ind_kf), np.array(ind_kt)]


class SimulatedPatientDatabase:
    def __init__(self, obs_type: 'TimeSeriesObservation', num_patients):

        self.obs_type = obs_type
        self.num_patients = num_patients
        self._init_data_structure()

        for i in range(self.num_patients):
            self.create_patient_data()

    def _init_data_structure(self):
        """
        Initialize data structure to hold time series data for all patient

        all_measurement: all training measurements for the time series
        all_response_progression: all training response for the time series

        """
        self.all_measurements = []
        self.all_response_progression = []

    def create_patient_data(self):
        """ Simulate patient's time sereis measurement and response """
        raise NotImplementedError


class SimulatedPatientDatabaseContinous(SimulatedPatientDatabase):
    def __init__(self, **kwargs):
        super(SimulatedPatientDatabaseContinous, self).__init__(**kwargs)

    def create_patient_data(self):
        pass


class SimulatedPatientDatabaseDiscrete(SimulatedPatientDatabase):
    """ Simulate patient data base that hold time series data """

    def __init__(self, obs_type: 'Disease', **kwargs):

        super(SimulatedPatientDatabaseDiscrete, self).__init__(**{'obs_type': obs_type, **kwargs})

    def _init_data_structure(self):

        super(SimulatedPatientDatabaseDiscrete, self)._init_data_structure()

        self.all_labels = []
        self.all_rel_end_time = []

    def create_patient_data(self):
        label, response_progression, measurements = \
            self.obs_type.create_patient_data()

        if label is not None:
            self.all_labels.append(label)

        self.all_response_progression.append(response_progression)
        self.all_measurements.append(measurements)
        if isinstance(measurements, list) and len(measurements) == 3:  # sparse matrix
            self.all_rel_end_time.append(self.obs_type.x[int(measurements[2][-1])])

    def _split_dataset(self, val_frac):
        """
        Return the the ind ids for train set and validation set
        """
        train_test_perm = np.random.permutation(self.num_patients)
        tr_inds = train_test_perm[int(val_frac * self.num_patients):]
        val_inds = train_test_perm[:int(val_frac * self.num_patients)]

        return tr_inds, val_inds

    def dump_to_loca(self, fname, val_frac=0.3):
        """
        Dump the simulated data into Kingsle's required format

        """
        def packaging(inds):
            # initialize data structure
            keys = ['Ys', 'Ts', 'ind_kts', 'ind_kfs', 'labels', 'covs',
                    'rel_end_time', 'feature_names', 'num_features', 'covs_names']
            data_dict = {k: [] for k in keys}

            # fill in data into the data structure
            data_dict['num_features'] = self.obs_type.num_features

            for ind in inds:
                data_dict['Ts'].append(self.obs_type.x)
                data_dict['Ys'].append(self.all_measurements[ind][0])
                data_dict['ind_kfs'].append(self.all_measurements[ind][1])
                data_dict['ind_kts'].append(self.all_measurements[ind][2])
                data_dict['labels'].append(self.all_labels[ind])
                data_dict['rel_end_time'].append(self.all_rel_end_time[ind])

            print('Num of patients die: {}'.format(np.mean(data_dict['labels'])))

            return data_dict

        tr_inds, val_inds = self._split_dataset(val_frac=val_frac)

        pickle.dump({'train': packaging(tr_inds), 'test': packaging(val_inds)}, open(fname, "wb"))
