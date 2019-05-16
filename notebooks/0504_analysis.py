import pickle
from utils.plotting import plot_measurement, plot_hist_simple
import time
from datetime import datetime as dt
import pandas as pd


def get_diff_time_by_hours(time1, time2):
    '''
    time1: '2018-02-11 16:33:43'
    time2: '2018-02-12 16:33:43'
    '''
    # Handle '2123-04-13' does not match format '%Y-%m-%d %H:%M:%S'
    if len(time1) == 10:
        time1 += ' 00:00:00'
    if len(time2) == 10:
        time2 += ' 00:00:00'

    a = dt.strptime(time1, '%Y-%m-%d %H:%M:%S')
    b = dt.strptime(time2, '%Y-%m-%d %H:%M:%S')

    diff = b - a
    hours = diff.days * 24
    hours += diff.seconds / 3600.
    return hours



def get_sparse_coding_from_dataset(data_dir='./data/mingjie-mortality/',
                                   mode='train',
                                   max_encounters=-1,
                                   max_hours=72):
    list_df = pd.read_csv('{}/{}/listfile.csv'.format(data_dir, mode))

    Ts = []
    Ys = []
    ind_ts = []
    ind_fs = []
    valid_encounter_idx = []

    if max_encounters == -1:
        max_encounters = len(list_df)

    # Collect spase coding
    for i in range(max_encounters):
        Y = []
        ind_t = []
        ind_f = []

        start_time = time.time()

        filename = list_df.iloc[i]['Filename']
        intime = list_df.iloc[i]['Intime']
        deathtime = list_df.iloc[i]['Deathtime']
        outtime = list_df.iloc[i]['Outtime']

        rel_end_time = get_diff_time_by_hours(intime, outtime)

        if pd.notnull(deathtime):
            rel_deathtime = get_diff_time_by_hours(intime, deathtime)
            rel_end_time = min(rel_deathtime, rel_end_time)

        measurements = pd.read_csv('{}/{}/{}'.format(
            data_dir, mode, filename
        ))

        measurements = measurements[(measurements['Hours'] >= 0) &
                                    (measurements['Hours'] <= rel_end_time)]

        T = measurements['Hours'].tolist()
        for j, j_name in enumerate(list(measurements)[2:]):
            valid_measurements = measurements[pd.notnull(measurements[j_name])]
            if len(valid_measurements) > 0:
                Y += valid_measurements[j_name].tolist()
                ind_f += [j] * len(valid_measurements)
                ind_t += valid_measurements.index.tolist()

        if len(ind_t) > 0:
            Ts.append(T)
            Ys.append(Y)
            ind_ts.append(ind_t)
            ind_fs.append(ind_f)
            valid_encounter_idx.append(i)

        if i % 1000 == 0:
            print('Finished {} / {}. Take {}s'.format(
                i, len(list_df), time.time() - start_time))
    # Collect time-invariant features
    list_df = list_df[valid_encounter_idx]
    return Ts, Ys, ind_ts, ind_fs



# def summarize_patient_data(t, y, ind_kt, ind_kf):
#     data = sparse_matrix_to_list(t, y, ind_kt, ind_kf)
#     num_features = max(ind_kf) + 1
#
#     summary = np.zeros(2 * num_features + 1)
#     assert len(y) == len(ind_kt) and len(y) == len(ind_kf)
#     summary[0] = len(y)
#     summary[1:num_features + 1] = [len(i[1]) for i in data]
#     summary[num_features + 1:2 * num_features + 1] = [sum(i[1]) for i in data]
#
#     return summary


def report_label_info(labels):
    num_samples = labels.shape[0]
    prop_cases = np.mean(labels)
    num_cases = np.sum(labels)

    print('Case/Control: {}/{}, {:.4f}/{:.4f}'.format(num_cases, num_samples - num_cases,
                                                      prop_cases, 1 - prop_cases))


if __name__ == '__main__':

    data_dir = '../data/mingjie-mortality/'

    # train_Ts, proc_train_Ys, proc_train_ind_ts, proc_train_ind_fs, train_covs, \
    # train_labels, train_rel_end_times = \
    #     pickle.load(open(data_dir + 'proc_train.pkl', 'rb'))

    test_Ts, proc_test_Ys, proc_test_ind_ts, proc_test_ind_fs, test_covs, \
    test_labels, test_rel_end_times = \
        pickle.load(open(data_dir + 'proc_test.pkl', 'rb'))

    # proc_aggregators = {field: [] for field in x.columns.tolist()[2:]}
    # for Ys, ind_fs in zip(proc_train_Ys + proc_test_Ys,
    #                       proc_train_ind_fs + proc_test_ind_fs):  # per encounter
    #     for Y, ind_f in zip(Ys, ind_fs):
    #         feature_name = idx_to_feature_name[ind_f]
    #         proc_aggregators[feature_name].append(Y)
    #
    # for k in proc_aggregators:
    #     values = np.array(proc_aggregators[k])
    #     plot_hist_simple(values, title='Hist of {}'.format(k))
    #     print(pd.DataFrame({k: values}).describe())

    plot_measurement(test_Ts[3], proc_test_Ys[3], proc_test_ind_ts[3], proc_test_ind_fs[3], test_Ts[3], y_upper=None, y_lower=None)

    print('done')