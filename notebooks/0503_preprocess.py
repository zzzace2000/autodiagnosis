import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle
from utils.plotting import plot_box_simple, plot_hist_simple, plot_scatter_simple

import time
from datetime import datetime as dt
import re


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


def transform_measurement_to_string(T, Y, ind_t, ind_f, negative_allow=False):
    invalid_idx = []
    num_measurements = len(Y)
    invalid_measurements = []

    for i, v in enumerate(Y):
        if type(v) == float or type(v) == int:
            if not negative_allow and Y[i] < 0:
                invalid_idx.append(i)
                invalid_measurements.append(v)
            continue
        try:
            Y[i] = float(re.search('(-?\d+)', v).group(1))
            if not negative_allow and Y[i] < 0:
                invalid_idx.append(i)
                invalid_measurements.append(v)
        except:
            invalid_idx.append(i)
            invalid_measurements.append(v)

    if len(invalid_idx) > 0:
        #T = [T[i] for i in range(num_measurements) if i not in invalid_idx]
        Y = [Y[i] for i in range(num_measurements) if i not in invalid_idx]
        ind_t = [ind_t[i] for i in range(num_measurements) if i not in invalid_idx]
        ind_f = [ind_f[i] for i in range(num_measurements) if i not in invalid_idx]

    return T, Y, ind_t, ind_f, invalid_measurements


def get_sparse_coding_from_dataset(data_dir='./data/mingjie-mortality/',
                                   mode='train',
                                   max_encounters=-1):
    list_df = pd.read_csv('{}/{}/listfile.csv'.format(data_dir, mode))

    Ts = []
    Ys = []
    ind_ts = []
    ind_fs = []
    valid_encounter_idx = []
    rel_end_times = []
    invalid_measurements = []
    start_time = time.time()

    if max_encounters == -1:
        max_encounters = len(list_df)

    for i in range(max_encounters):
        Y = []
        ind_t = []
        ind_f = []

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
        del measurements['Height']

        T = measurements['Hours'].tolist()
        measurements = measurements[(measurements['Hours'] >= 0) &
                                    (measurements['Hours'] <= rel_end_time)]

        for j, j_name in enumerate(list(measurements)[2:]):
            valid_measurements = measurements[pd.notnull(measurements[j_name])]
            if len(valid_measurements) > 0:
                Y += valid_measurements[j_name].tolist()
                ind_f += [j] * len(valid_measurements)
                ind_t += valid_measurements.index.tolist()

        if len(ind_t) > 0:
            T, Y, ind_t, ind_f, invalid_measurement = transform_measurement_to_string(T, Y, ind_t, ind_f)
            invalid_measurements += invalid_measurement

        if len(ind_t) > 0:
            rel_end_times.append(rel_end_time)
            Ts.append(T)
            Ys.append(Y)
            ind_ts.append(ind_t)
            ind_fs.append(ind_f)
            valid_encounter_idx.append(i)

        if i == max_encounters - 1 or (i % 4000 == 0 and i != 0):
            for v in set(invalid_measurements):
                print('Invalid measurement  {}'.format(v))
            print('Finished {} / {}. Take {:.2f}s'.format(
                i, len(list_df), time.time() - start_time))
            start_time = time.time()
            invalid_measurements = []

    return valid_encounter_idx, Ts, Ys, ind_ts, ind_fs, rel_end_times


def get_one_hot_encoding(df, cname):
    df = pd.concat([df, pd.get_dummies(df[cname], prefix=cname)], axis=1)
    df.drop([cname], axis=1, inplace=True)
    return df


def get_time_invariant_features(data_dir='./data/mingjie-mortality/',
                                mode='train',
                                valid_encounter_idx=None,
                                time_invariant_fields=None):
    list_df = pd.read_csv('{}/{}/listfile.csv'.format(data_dir, mode))
    if time_invariant_fields is not None:
        list_df = list_df[time_invariant_fields]

    if valid_encounter_idx is not None:
        list_df = list_df.loc[valid_encounter_idx]

    list_df = get_one_hot_encoding(list_df, 'Ethnicity')

    return list_df


def get_labels(data_dir='./data/mingjie-mortality/',
               mode='train',
               valid_encounter_idx=None):
    list_df = pd.read_csv('{}/{}/listfile.csv'.format(data_dir, mode))
    if valid_encounter_idx is not None:
        list_df = list_df.loc[valid_encounter_idx]
    return list_df['Mortality'].tolist()


def adjust_measurements(Ys, ind_ts, ind_fs, stat_dict, idx_to_feature_name, log_fields):
    new_Ys = []
    new_ind_ts = []
    new_ind_fs = []
    max_encounters = len(Ys)
    start_time = time.time()

    for i in range(max_encounters):
        new_Y = []
        new_ind_t = []
        new_ind_f = []

        for Y, ind_t, ind_f in zip(Ys[i], ind_ts[i], ind_fs[i]):
            feature_name = idx_to_feature_name[ind_f]
            stat = stat_dict[feature_name]
            if stat['min'] <= Y <= stat['max']:
                if feature_name in log_fields:
                    Y = np.log(Y + 0.1)

                Y -= stat['mean']
                if stat['std'] != 0:
                    Y /= stat['std']

                new_Y.append(Y)
                new_ind_t.append(ind_t)
                new_ind_f.append(ind_f)

        new_Ys.append(new_Y)
        new_ind_ts.append(new_ind_t)
        new_ind_fs.append(new_ind_f)

        if i == max_encounters - 1 or (i % 4000 == 0 and i != 0):
            print('Finished {} / {}. Take {:.2f}s'.format(
                i, max_encounters, time.time() - start_time))
            start_time = time.time()

    return new_Ys, new_ind_ts, new_ind_fs


def play_with_one_time_series(data_dir):
    df = pd.read_csv(data_dir + 'train/3_episode1_timeseries.csv')
    del df['Height']
    print(df.head(5))
    plot_scatter_simple(df['Hours'], df['Temperature'], title='Hours vs Temperature')

    return df


def play_with_all_patient_record(data_dir):
    time_invariant_fields = ['Age', 'Ethnicity', 'Gender']
    train_ids, train_Ts, train_Ys, train_ind_ts, train_ind_fs, \
    train_rel_end_times = \
        get_sparse_coding_from_dataset(data_dir=data_dir,
                                       mode='train',
                                       max_encounters=-1)

    test_ids, test_Ts, test_Ys, test_ind_ts, test_ind_fs, \
    test_rel_end_times = \
        get_sparse_coding_from_dataset(data_dir=data_dir,
                                       mode='test',
                                       max_encounters=-1)

    train_covs = get_time_invariant_features(data_dir=data_dir,
                                             mode='train',
                                             valid_encounter_idx=train_ids,
                                             time_invariant_fields=time_invariant_fields)
    test_covs = get_time_invariant_features(data_dir=data_dir,
                                            mode='test',
                                            valid_encounter_idx=test_ids,
                                            time_invariant_fields=time_invariant_fields)
    all_covs = pd.concat([train_covs, test_covs])
    all_covs['Age'] = (all_covs['Age'] - np.mean(all_covs['Age'])) / np.std(all_covs['Age'])
    all_covs['Gender'] = (all_covs['Gender'] - np.mean(all_covs['Gender'])) / np.std(all_covs['Gender'])

    train_covs = all_covs[:len(train_covs)]
    test_covs = all_covs[len(train_covs):]

    plot_hist_simple(train_covs['Age'], "Age")
    plot_hist_simple(train_covs['Gender'], 'Gender')

    train_labels = get_labels(data_dir=data_dir,
                              mode='train',
                              valid_encounter_idx=train_ids)
    test_labels = get_labels(data_dir=data_dir,
                             mode='test',
                             valid_encounter_idx=test_ids)

    pickle.dump((train_Ts, train_Ys, train_ind_ts, train_ind_fs, train_covs,
                 train_labels, train_rel_end_times),
                open(data_dir + 'raw_train.pkl', 'wb'))
    pickle.dump((test_Ts, test_Ys, test_ind_ts, test_ind_fs, test_covs,
                 test_labels, test_rel_end_times),
                open(data_dir + 'raw_test.pkl', 'wb'))
    return


def deal_with_data_correction(data_dir, x):

    train_Ts, train_Ys, train_ind_ts, train_ind_fs, train_covs, \
    train_labels, train_rel_end_times = \
        pickle.load(open(data_dir + 'raw_train.pkl', 'rb'))

    test_Ts, test_Ys, test_ind_ts, test_ind_fs, test_covs, \
    test_labels, test_rel_end_times = \
        pickle.load(open(data_dir + 'raw_test.pkl', 'rb'))

    binary_fields = ['Capillary refill rate']
    idx_to_feature_name = dict([(idx, name) for idx, name in enumerate(x.columns[2:])])
    aggregators = {field: [] for field in x.columns.tolist()[2:]}

    for Ys, ind_fs in zip(train_Ys + test_Ys, train_ind_fs + test_ind_fs):  # per encounter
        for Y, ind_f in zip(Ys, ind_fs):
            feature_name = idx_to_feature_name[ind_f]
            aggregators[feature_name].append(Y)

    stat_dict = {}

    for k in aggregators:
        values = aggregators[k]
        if k in binary_fields:
            aggregators[k] = values
            continue
        values = np.array(values)

        IQR = np.percentile(values, 75) - np.percentile(values, 25)
        median = np.percentile(values, 50)

        values = values[values <= (median + 4.5 * IQR)]
        values = values[values >= (median - 4.5 * IQR)]
        aggregators[k] = values

    for k in aggregators:
        values = np.array(aggregators[k])
        stat_dict[k] = {'max': np.max(values),
                        'min': np.min(values)}

    log_fields = ['Bilirubin', 'Blood urea nitrogen', 'Creatinine', 'Cholesterol',
                  'Glascow coma scale total', 'Lactate', 'Oxygen saturation',
                  'Partial pressure of carbon dioxide', 'Partial pressure of oxygen',
                  'Partial thromboplastin time', 'Platelets', 'Urine output',
                  'White blood cell count', 'pH']

    for k in aggregators:
        values = aggregators[k]
        if k in log_fields:
            aggregators[k] = np.log(values + 0.1)

    for k in aggregators:
        values = np.array(aggregators[k])
        stat_dict[k]['mean'] = np.mean(values)
        stat_dict[k]['std'] = np.std(values)

    proc_train_Ys, proc_train_ind_ts, proc_train_ind_fs = \
        adjust_measurements(train_Ys, train_ind_ts, train_ind_fs, stat_dict, idx_to_feature_name, log_fields)

    proc_test_Ys, proc_test_ind_ts, proc_test_ind_fs = \
        adjust_measurements(test_Ys, test_ind_ts, test_ind_fs, stat_dict, idx_to_feature_name, log_fields)


    pickle.dump((train_Ts, proc_train_Ys, proc_train_ind_ts, proc_train_ind_fs, train_covs,
                 train_labels, train_rel_end_times),
                open(data_dir + 'proc_train.pkl', 'wb'))
    pickle.dump((test_Ts, proc_test_Ys, proc_test_ind_ts, proc_test_ind_fs, test_covs,
                 test_labels, test_rel_end_times),
                open(data_dir + 'proc_test.pkl', 'wb'))


def put_stuff_in_dict(data_dir):
    train_Ts, train_Ys, train_ind_ts, train_ind_fs, train_covs, train_labels, train_rel_end_time = \
        pickle.load(open(data_dir + 'proc_train.pkl', 'rb'))
    test_Ts, test_Ys, test_ind_ts, test_ind_fs, test_covs, test_labels, test_rel_end_time = \
        pickle.load(open(data_dir + 'proc_test.pkl', 'rb'))

    feature_names = [
        'Albumin', 'Bicarbonate', 'Bilirubin', 'Blood urea nitrogen',
        'CO2', 'Calcium', 'Calcium ionized', 'Capillary refill rate',
        'Chloride', 'Cholesterol', 'Creatinine', 'Diastolic blood pressure',
        'Fraction inspired oxygen', 'Glascow coma scale total', 'Glucose',
        'Heart Rate', 'Hemoglobin', 'Lactate', 'Magnesium',
        'Mean blood pressure', 'Oxygen saturation',
        'Partial pressure of carbon dioxide', 'Partial pressure of oxygen',
        'Partial thromboplastin time', 'Platelets', 'Potassium',
        'Prothrombin time', 'Respiratory rate', 'Systolic blood pressure',
        'Temperature', 'Urine output', 'Weight', 'White blood cell count',
        'pH']

    result_dict = {}
    result_dict['train'] = dict(Ys=train_Ys, Ts=train_Ts, ind_kts=train_ind_ts,
                                ind_kfs=train_ind_fs, labels=train_labels, covs=train_covs.values,
                                rel_end_time=train_rel_end_time,
                                feature_names=feature_names, num_features=len(feature_names),
                                covs_names=list(train_covs.columns))
    result_dict['test'] = dict(Ys=test_Ys, Ts=test_Ts, ind_kts=test_ind_ts,
                               ind_kfs=test_ind_fs, labels=test_labels, covs=test_covs.values,
                               rel_end_time=test_rel_end_time,
                               feature_names=feature_names,
                               num_features=len(feature_names),
                               covs_names=list(train_covs.columns))

    pickle.dump(result_dict, open(data_dir + 'mingie_34features.pkl', 'wb'))

if __name__ == '__main__':
    data_dir = '../data/mingjie-mortality/'

    # x = play_with_one_time_series(data_dir)
    # play_with_all_patient_record(data_dir, x)
    # deal_with_data_correction(data_dir)

    put_stuff_in_dict(data_dir)



