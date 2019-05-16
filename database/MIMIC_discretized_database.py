import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from database.MIMIC_discretized_exp import MIMIC_discretized_exp
import numpy as np


class MIMIC_discretized_database(MIMIC_discretized_exp):
    """ Deal with sparse array of discriteized MIMIC data """
    def __init__(self, RL_interval, data_dir, database_name='mingie_34features.pkl',
                 num_hours_warmup=12, min_measurements_in_warmup=10,
                 num_hours_pred=6, num_X_pred=25, val_ratio=0.1, num_gp_samples_limit=1000,
                 include_before_death=48, verbose=True):

        super(MIMIC_discretized_database, self).__init__(
            RL_interval=RL_interval, reward_func=None, mgp_rnn=None,
            data_dir=data_dir, database_name=database_name, num_hours_warmup=num_hours_warmup,
            min_measurements_in_warmup=min_measurements_in_warmup,
            num_hours_pred=num_hours_pred,
            num_X_pred=num_X_pred,
            val_ratio=val_ratio,
            num_gp_samples_limit=num_gp_samples_limit,
            include_before_death=include_before_death,
            verbose=verbose)

    def create_loaders(self, batch_size):
        # Training set is the positive case with subsampled neg case
        # num_pos = self.pos_train_idxes.shape[0]
        # neg_idxes = np.random.choice(self.neg_train_idxes, num_pos)
        # total_train_idxes = np.concatenate([self.pos_train_idxes, neg_idxes])

        total_train_idxes = self.pos_train_idxes

        np.random.shuffle(total_train_idxes)

        train_loader = self._gen_pat_data_loader(batch_size, total_train_idxes, self.train_dict, num_pat_produced=3)
        val_loader = self._gen_pat_data_loader(batch_size, self.pos_val_idxes, self.train_dict, num_pat_produced=3)

        return train_loader, val_loader

    def create_test_loader(self, batch_size):
        return self._gen_pat_data_loader(batch_size, self.pos_test_idxes, self.test_dict, num_pat_produced=3)

    def _gen_pat_data_loader(self, batch_size, patient_idxes, the_dict, num_pat_produced=2):
        yield len(patient_idxes)

        loader = super(MIMIC_discretized_database, self)._gen_pat_data_loader(
                  batch_size, patient_idxes, the_dict, num_pat_produced)
        for tmp in loader:
            yield tmp

    def _collect_the_data(self, result_dict, f_Y, f_ind_kf, f_ind_kt, f_T,
                          s_Y, s_ind_kf, s_ind_kt, s_T, new_X, new_labels, patient_idx,
                          next_idx, warmup_idx, interval_times,
                          cov=None):
        '''
        Modify the running experience to train MGP_RNN. But we want a joint action examples here.
        '''

        new_s_ind_kt = [len(f_T) + the_ind_kt for the_ind_kt in s_ind_kt]

        result_dict['Ys'].append(f_Y + s_Y)
        result_dict['ind_kfs'].append(f_ind_kf + s_ind_kf)
        result_dict['ind_kts'].append(f_ind_kt + new_s_ind_kt)
        result_dict['Ts'].append(f_T + s_T)
        result_dict['Xs'].append(new_X)
        result_dict['labels'].append(new_labels)
        if cov is not None:
            result_dict['covs'].append(cov)



