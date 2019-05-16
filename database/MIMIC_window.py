import pickle
import numpy as np
import os
import time
import sys

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from database.Exceptions import InvalidPatientException


class MIMIC_window(object):
    def __init__(self, data_dir=None, database_name='mingie_34features',
                 num_hours_warmup=0, min_measurements_in_warmup=3,
                 num_hours_pred=6, num_X_pred=25, X_interval=1, val_ratio=0.1,
                 num_gp_samples_limit=None,
                 include_before_death=49, before_end=0., data_interval=12,
                 verbose=True, num_pat_produced=10, cache_dir=None,
                 neg_subsampled=False, **kwargs):
        '''
        Generate experience for classifier_training_and_evaluation.
        :param mgp_rnn: the pretrained classifier
        :param data_dir: the data directory
        :param database_name: choose btw ['norm_10features_hard.pkl.pkl', 'norm_34features_hard.pkl']

        :param num_hours_warmup: At least these number of hours ahead of time to impute.
        :param num_hours_pred: Forward prediction these num of hours.
        :param num_X_pred: Max time points feeding into RNN to predict (24)
        '''
        self.data_dir = data_dir
        self.database_name = database_name

        self.num_hours_warmup = num_hours_warmup
        self.min_measurements_in_warmup = min_measurements_in_warmup
        self.num_hours_pred = num_hours_pred
        self.num_X_pred = num_X_pred
        self.X_interval = X_interval
        self.val_ratio = val_ratio
        self.num_gp_samples_limit = num_gp_samples_limit
        self.include_before_death = include_before_death
        self.before_end = before_end
        self.data_interval = data_interval
        self.num_pat_produced = num_pat_produced
        self.neg_subsampled = neg_subsampled
        print('neg_subsampled:', neg_subsampled)

        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cache = (cache_dir is not None)

        if not self.use_cache:
            self._get_data()

    def _get_data(self):
        with open(os.path.join(self.data_dir, '%s.pkl' % self.database_name), 'rb') as fp:
            self.database = pickle.load(fp)

        self.feature_names = self.database['feature_names']
        self.num_features = len(self.feature_names)

        self.database['labels'] = np.array(self.database['labels'], dtype=int)

        self.pos_train_idxes = np.arange(self.database['labels'].shape[0])[
            (self.database['labels'] == 1) & (self.database['test_indicators'] == False)]
        self.neg_train_idxes = np.arange(self.database['labels'].shape[0])[
            (self.database['labels'] == 0) & (self.database['test_indicators'] == False)]

        # Random split train and validation
        def split_idxes_to_train_and_val(idxes):
            np.random.shuffle(idxes)
            thelen = int(self.val_ratio * len(idxes))
            if thelen == 0:
                thelen = 1
            pos_val_idxes = idxes[:thelen]
            pos_train_idxes = idxes[thelen:]
            return pos_train_idxes, pos_val_idxes

        self.pos_train_idxes, self.pos_val_idxes = split_idxes_to_train_and_val(self.pos_train_idxes)
        self.neg_train_idxes, self.neg_val_idxes = split_idxes_to_train_and_val(self.neg_train_idxes)

        self.train_idxes = np.random.permutation(np.concatenate((self.pos_train_idxes, self.neg_train_idxes), axis=0))
        self.val_idxes = np.random.permutation(np.concatenate((self.pos_val_idxes, self.neg_val_idxes), axis=0))
        self.test_idxes = self.database['test_set_idxes']

        self.pos_test_idxes = np.arange(self.database['labels'].shape[0])[
            (self.database['labels'] == 1) & (self.database['test_indicators'])]

    @classmethod
    def load_database_from_trained_dir(cls, model_dir):
        hyper_path = os.path.join(model_dir, 'hyperparams.log')
        if not os.path.exists(hyper_path):
            raise FileNotFoundError('No file found in the path %s' % hyper_path)

        params = json.load(open(hyper_path, 'rb'))

        database = cls(**params['database'])
        return database

    @staticmethod
    def load_database_by_category(data_dir, database_name, **kwargs):
        if database_name == 'simulated_6features':
            params = dict(
                data_dir=data_dir,
                database_name=database_name,
                num_gp_samples_limit=300,
                data_interval=0.34, X_interval=0.34,
                before_end=0.,
                num_hours_warmup=2, min_measurements_in_warmup=0,
                num_hours_pred=0.7, num_X_pred=17, val_ratio=0.3,
                include_before_death=5, verbose=False, neg_subsampled=False)
        elif database_name == 'simulated_6features_big_mean':
            params = dict(
                data_dir=data_dir,
                database_name=database_name,
                num_gp_samples_limit=300,
                data_interval=0.25, X_interval=0.25,
                before_end=0.,
                num_hours_warmup=2, min_measurements_in_warmup=0,
                num_hours_pred=0.75, num_X_pred=17, val_ratio=0.3,
                include_before_death=5, verbose=False, neg_subsampled=False)
        else:
            params = dict(
                data_dir=data_dir,
                database_name=database_name,
                num_gp_samples_limit=300,
                data_interval=5.5, X_interval=1,
                before_end=0.25,
                num_hours_warmup=6, min_measurements_in_warmup=10,
                num_hours_pred=6, num_X_pred=25, val_ratio=0.1,
                include_before_death=30, verbose=False, neg_subsampled=False)
        params.update(kwargs)
        new_patient_database = MIMIC_window(**params)

        return new_patient_database, params

    def create_loaders(self, batch_size):
        total_train_idxes = self.train_idxes
        if self.neg_subsampled and len(self.neg_train_idxes) > 0:
            # Training set is the positive case with subsampled neg case
            neg_idxes = np.random.choice(self.neg_train_idxes, self.pos_train_idxes.shape[0])
            total_train_idxes = np.concatenate([self.pos_train_idxes, neg_idxes])
            np.random.shuffle(total_train_idxes)

        train_loader = self._gen_pat_data_loader(batch_size, total_train_idxes)
        val_loader = self._gen_pat_data_loader(batch_size, self.val_idxes)
        return train_loader, val_loader

    def create_test_loader(self, batch_size):
        return self._gen_pat_data_loader(
            batch_size=batch_size, patient_idxes=self.test_idxes)

    def _gen_pat_data_loader(self, batch_size, patient_idxes, transform=None):
        '''
        Create a loader that outputs patients information to get rewards.
        # Handle actions that have same time:
        # - Take previous all measurements as 1st array
        # - Take the same time point as 2nd array, then loop through them to construct experiences
        # - Then concatenate 1st and 2nd array as 1st array. Construct 2nd array
        # - Till end of array
        '''
        if patient_idxes is None:
            patient_idxes = range(len(self.database['Ys']))
        num_patients = len(patient_idxes)
        num_skip_patients = 0

        yield num_patients

        attributes = ['Ys', 'Ts', 'ind_kfs', 'ind_kts', 'labels']
        addi_attributes = ['time_to_ends', 'Xs', 'patient_inds', 'the_steps', 'total_steps']
        if 'covs' in self.database:
            addi_attributes.append('covs')
        if 'missings' in self.database:
            addi_attributes.append('missings')

        result_dict = dict([(attr, []) for attr in attributes + addi_attributes])

        for p_i, patient_idx in enumerate(patient_idxes):
            Y, T, ind_kf, ind_kt, label, episode_end_time = \
                (self.database[attr][patient_idx] for attr in attributes + ['rel_end_time'])

            new_Y, new_T, new_ind_kf, new_ind_kt = Y, T, ind_kf, ind_kt
            if transform is not None:
                new_Y, new_T, new_ind_kf, new_ind_kt = transform(Y, T, ind_kf, ind_kt,
                                                                 episode_end_time)

            cov = None
            if 'covs' in self.database and len(self.database['covs']) > 0:
                cov = self.database['covs'][patient_idx]

            try:
                start_ind, end_ind = self._is_valid(new_Y, new_T, new_ind_kf,
                                                    new_ind_kt, label, episode_end_time)
            except InvalidPatientException as e:
                num_skip_patients += 1
                print(e)
                continue

            # Record things
            ind_arr_loader = self._gen_ind_arr_loader(start_ind, end_ind,
                                                      episode_end_time, new_T, new_ind_kt)

            for ind_arr_result in ind_arr_loader:
                # Add stuff into result dict
                self._record_results_by_idxes(ind_arr_result, new_Y, new_T, new_ind_kt, new_ind_kf,
                                              label, episode_end_time, result_dict, patient_idx,
                                              cov)

            # if p_i % 1 == 0:
            #     if len(result_dict['Ys']) < 2:
            #         self._verbose_print('Discard: less than 2 data')
            #         del result_dict
            #         result_dict = dict([(attr, []) for attr in attributes + addi_attributes])
            #         continue

            # Produce batches of experiences
            if p_i % self.num_pat_produced == (self.num_pat_produced - 1) or p_i == len(patient_idxes) - 1:
                self._verbose_print('[{}/{}] Produced {} data.'.format(
                    p_i, num_patients, len(result_dict['Ys'])))

                if isinstance(result_dict['labels'], list):
                    result_dict['labels'] = np.expand_dims(np.array(result_dict['labels'], dtype=int), -1)

                if batch_size is None or batch_size >= len(result_dict['Ys']):
                    yield result_dict
                else:
                    for batch_idx in range(0, len(result_dict['Ys']), batch_size):
                        ret_dict = {k: result_dict[k][batch_idx:(batch_idx + batch_size)] for k in result_dict}
                        yield ret_dict

                del result_dict
                result_dict = dict([(attr, []) for attr in attributes + addi_attributes])

        self._verbose_print('Skip %d patients out of total %d patients' % (num_skip_patients, num_patients))

    def _is_valid(self, Y, T, ind_kf, ind_kt, label, episode_end_time):
        if self.min_measurements_in_warmup >= len(ind_kt):
            error_msg = 'Only have %d measurements but required min %d number. Discard.' % \
                        (len(ind_kt), self.min_measurements_in_warmup)
            raise InvalidPatientException(error_msg)

        if T[ind_kt[-1]] > episode_end_time:
            raise InvalidPatientException('Patient has measurements after the end time...')

        num_hours_warmup = max(T[ind_kt[self.min_measurements_in_warmup]], self.num_hours_warmup)

        if T[ind_kt[-1]] < num_hours_warmup:
            error_msg = 'Only have %.2f hours but minimum warmup time is %d. Discard.'  \
                        % (T[ind_kt[-1]], num_hours_warmup)
            raise InvalidPatientException(error_msg)

        # Get which index that it will pass the warmup time
        # warm up time is determined by min_measurements_in_warmup and num_hours_warmup

        start_ind = self.min_measurements_in_warmup
        while start_ind < (len(ind_kt) - 1) and T[ind_kt[start_ind]] < num_hours_warmup:
            start_ind += 1

        # Take the experiences before the death time
        while self.include_before_death is not None and \
                T[ind_kt[start_ind]] < (episode_end_time - self.include_before_death):
            start_ind += 1
            if start_ind >= len(ind_kt):
                error_msg = 'No measurements within %d hours of dying' % \
                            self.include_before_death
                raise InvalidPatientException(error_msg)

        # Get end_ind
        end_ind = len(ind_kt) - 1
        while end_ind > 0 and \
                T[ind_kt[end_ind]] > episode_end_time - self.before_end:
            end_ind -= 1

        if end_ind < start_ind:
            raise InvalidPatientException('End index and start index has no overlap!')

        return start_ind, end_ind

    def _gen_ind_arr_loader(self, start_ind, end_ind, episode_end_time, T, ind_kt):
        '''
        Each time it yields an array of indexes of Y for a classification.
        It is determined based on how long is the RNN time sequence, or which interval we evaluate.
        It takes the grid points between [-include_before_death, -before_end] with interval data_interval.
        All the index here represents the index of Y, indicating what measurements are included
        :param start_ind: usually 0. Determined by min_measurements_in_warm_up
        :param end_ind: usually last index of measurements. Affected by before_end.
        :param episode_end_time: when the patient dies. Used to determine the grid point.
        :param T: All the time that measurement happens
        :param ind_kt: The index for T
        :return:
        '''
        start_time = 0
        if self.include_before_death is not None:
            start_time = max(0, episode_end_time - self.include_before_death)

        all_times = np.arange(episode_end_time - self.before_end, start_time,
                              -self.data_interval)[::-1]

        total_step = len(all_times)
        the_ind_t = start_ind

        for step, end_pred_time in enumerate(all_times):
            if end_pred_time < T[ind_kt[start_ind]]:
                continue

            while the_ind_t <= end_ind and T[ind_kt[the_ind_t]] < end_pred_time:
                the_ind_t += 1

            ind_arr = range(the_ind_t)

            result = dict(ind_arr=ind_arr, end_pred_time=end_pred_time, step=step, total_step=total_step)
            yield result

    def _record_results_by_idxes(self, ind_arr_result, Y, T, ind_kt, ind_kf,
                                 label, episode_end_time, result_dict, patient_idx,
                                 cov=None):
        ind_arr, end_pred_time, step, total_steps = (
            ind_arr_result[k] for k in ['ind_arr', 'end_pred_time', 'step', 'total_step'])

        if len(ind_arr_result['ind_arr']) == 0:
            print('bug here! the first arr should not be empty.')
            return

        if self.num_gp_samples_limit is not None and len(ind_arr_result['ind_arr']) > self.num_gp_samples_limit:
            ind_arr = ind_arr[-self.num_gp_samples_limit:]

        # Construct X
        start_time = max(end_pred_time - self.num_X_pred, 0)
        new_X = np.arange(end_pred_time, start_time, -self.X_interval)[::-1]
        new_labels = 0 if label == 0 else int(end_pred_time + self.num_hours_pred >= episode_end_time)
        new_time_to_end = episode_end_time - end_pred_time

        offset = ind_kt[ind_arr[0]]

        new_Y = [Y[idx] for idx in ind_arr]
        new_ind_kt = [ind_kt[idx] - offset for idx in ind_arr]
        new_ind_kf = [ind_kf[idx] for idx in ind_arr]
        new_T = T[offset:(new_ind_kt[-1] + 1 + offset)]

        result_dict['Ys'].append(new_Y)
        result_dict['ind_kts'].append(new_ind_kt)
        result_dict['ind_kfs'].append(new_ind_kf)
        result_dict['Ts'].append(new_T)
        result_dict['Xs'].append(new_X)
        result_dict['labels'].append(new_labels)
        result_dict['time_to_ends'].append(new_time_to_end)

        result_dict['patient_inds'].append(patient_idx)
        result_dict['the_steps'].append(step)
        result_dict['total_steps'].append(total_steps)
        if cov is not None:
            result_dict['covs'].append(cov)

    def _verbose_print(self, the_str):
        if self.verbose:
            print(the_str)


class MIMIC_discretized_window(MIMIC_window):
    '''
    Generate experience for each discredited time point, alisgned by the last time point

    '''
    def __init__(self, dis_interval, **kwargs):
        super(MIMIC_discretized_window, self).__init__(**kwargs)
        self.dis_interval = dis_interval

    def _gen_pat_data_loader(self, **kwargs):
        return super(MIMIC_discretized_window, self)._gen_pat_data_loader(
            transform=self._discretize, **kwargs)

    def _discretize(self, Y, T, ind_kf, ind_kt, episode_end_time):
        # Discretize the measurments. Aligned by the last time point
        interval_times = np.arange(episode_end_time, 0, -self.dis_interval)[::-1]

        sum_measurements = np.zeros((self.num_features, len(interval_times)))
        count_measurements = np.zeros((self.num_features, len(interval_times)))

        # Loop through all the measurements
        for y, ind_f, ind_t in zip(Y, ind_kf, ind_kt):
            if T[ind_t] < (interval_times[0] - self.dis_interval / 2):
                continue
            the_interval = int((T[ind_t] - interval_times[0] - self.dis_interval / 2)
                               // self.dis_interval)

            sum_measurements[ind_f, the_interval] += y
            count_measurements[ind_f, the_interval] += 1

        measurements = np.divide(sum_measurements, count_measurements,
                                 out=np.zeros_like(sum_measurements),
                                 where=(count_measurements != 0))

        # Transform back to sparse arr
        new_Y, new_ind_kt, new_ind_kf = [], [], []
        for col in range(len(interval_times)):
            for feature_idx in range(self.num_features):
                if count_measurements[feature_idx, col] != 0:
                    new_Y.append(measurements[feature_idx, col])
                    new_ind_kf.append(feature_idx)
                    new_ind_kt.append(col)

        new_T = interval_times - self.dis_interval / 2
        return new_Y, new_T, new_ind_kf, new_ind_kt


if __name__ == '__main__':
    import tensorflow as tf
    import json
    from arch import ManyToOneMGP_RNN

    np.random.seed(0)
    tf.set_random_seed(0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        mgp_rnn = ManyToOneMGP_RNN.load_saved_mgp_rnn(
            sess, '../../models/0505-mimic-nh16-nl1-c1e-04-n_mc_15-nhrs6-'
                  'MIMIC_discretized_database-mingie_34features-ManyToOneMGP_RNN')

        print('Load model done!')

        database = MIMIC_window(
            data_interval=5, before_end=0.25, num_hours_pred=6,
            add_missing=True, dis_interval=0.25, data_dir='../../data/my-mortality/')

        test_loader = database.create_test_loader(batch_size=128)
        total_batches = next(test_loader)

        for obs_dict in test_loader:
            try:
                feed_dict = mgp_rnn.create_feed_dict_by_sparse_arr(**obs_dict)
                loss_fit_, loss_reg_, O_pred, probs, gp_length = sess.run([
                    mgp_rnn.loss_fit, mgp_rnn.loss_reg, mgp_rnn.O_pred, mgp_rnn.probs, mgp_rnn.length], feed_dict)
            except tf.errors.InvalidArgumentError as e:
                print(e)

            probs = probs[:, :, 0, 1]

            avg_probs = np.mean(probs, axis=1, keepdims=True)
            std_probs = np.std(probs, axis=1, keepdims=True)
            all_combined_probs = np.concatenate([avg_probs, std_probs], axis=1)
            labels = np.array(obs_dict['labels'])
            print('Done!')
