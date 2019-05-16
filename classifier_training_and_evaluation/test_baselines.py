from database.MIMIC_window import MIMIC_window
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import datetime
from arch.RNN import ManyToOneRNN
import os
import json
from utils.general import Timer


def get_x_and_y_from_loader(loader):
    total_batches = next(loader)
    the_func = getattr(ManyToOneRNN, args.rnn_imputation)

    x, y = [], []
    for obs_dict in loader:
        measurements, labels = the_func(
            args.num_features, args.add_missing, **obs_dict)
        print(measurements.shape)
        x.append(measurements.reshape(measurements.shape[0], -1))
        y.append(labels)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y.ravel()


def train_rf(database):
    train_loader, val_loader = database.create_loaders(batch_size=None)
    test_loader = database.create_test_loader(batch_size=None)

    with Timer('loading training set'):
        trainset = get_x_and_y_from_loader(train_loader)

    with Timer('training'):
        if args.classifier == 'RF':
            the_classifier = RandomForestClassifier(n_estimators=500, class_weight=args.class_weight)
        elif args.classifier == 'LR':
            the_classifier = LogisticRegression(C=1, class_weight=args.class_weight)
        else:
            raise NotImplementedError('Unknown classifier: %s' % args.classifier)
        the_classifier.fit(*trainset)

    with Timer('evaluating'):
        testset = get_x_and_y_from_loader(test_loader)

        x_test, y_test = testset
        x_pred_prob = the_classifier.predict_proba(x_test)[:, -1]

    # Calculate Metrics
    result = {}
    result['auroc'] = roc_auc_score(y_test, x_pred_prob)
    result['aupr'] = average_precision_score(y_test, x_pred_prob)
    result['acc'] = (np.round(x_pred_prob) == y_test).mean()
    result['positivity'] = ((np.round(x_pred_prob) == y_test)[y_test == 1]).mean()

    print(result)
    return result


def run():
    database_hyperparams = dict(
        data_dir=args.database_dir,
        database_name=args.database,
        num_gp_samples_limit=500,
        data_interval=args.data_interval, X_interval=args.X_interval,
        before_end=args.before_end,
        num_hours_warmup=args.num_hours_warmup, min_measurements_in_warmup=args.min_measurements_in_warmup,
        num_hours_pred=args.num_hours_pred, num_X_pred=args.num_X_pred, val_ratio=args.val_ratio,
        num_pat_produced=200,
        include_before_death=args.include_before_death, verbose=False, neg_subsampled=args.neg_subsampled
    )

    new_patient_database = MIMIC_window(**database_hyperparams)

    if args.mimic_cls != 'MIMIC_window':
        raise NotImplementedError('No MIMIC database found! %s' % args.mimic_cls)

    test_metrics = train_rf(new_patient_database)
    test_metrics.update(database_hyperparams)

    if not os.path.exists('./performance/'):
        os.mkdir('./performance/')

    the_file_name = '%s_%s_%s_baseline_miss%d_%s_nhrs%d.log' % (
        args.classifier, args.rnn_imputation, args.identifier, args.add_missing,
        args.database, args.num_hours_pred)
    with open('./performance/' + the_file_name, 'w') as op:
        json.dump(test_metrics, op)

    print('done')


def parse_args():
    parser = argparse.ArgumentParser(description='MGP_RNN')

    ## database
    parser.add_argument('--database_dir', type=str, default='../data/my-mortality/',
                        help='Where the dataset folder is')
    parser.add_argument('--database', type=str,
                        default='mingjie_39features_38covs',
                        help='The pickle file that must ends with .pkl. '
                             'E.g. Default file is in '
                             '../data/my-mortality/mingjie_39features_38covs.pkl')
    parser.add_argument('--num_hours_pred', type=float, default=24,
                        help='How many hours far to forecast if patient dies')
    parser.add_argument('--before_end', type=float, default=0.)
    parser.add_argument('--include_before_death', type=float, default=24.01)
    parser.add_argument('--data_interval', type=float, default=3,
                        help='The evaluation interval')
    parser.add_argument('--X_interval', type=float, default=1,
                        help='This specifies the time interval of RNN. Default is 1 hour.')
    parser.add_argument('--num_X_pred', type=float, default=24,
                        help='This specifies the number of point feeding into RNN.')
    parser.add_argument('--num_hours_warmup', type=float, default=3,
                        help='The time series needs to be at least these length')
    parser.add_argument('--min_measurements_in_warmup', type=int, default=5,
                        help='It needs to have at least 5 measurements in the warmup period')
    # Other data settings
    parser.add_argument('--neg_subsampled', type=int, default=1,
                        help='Subsample negative data to have the same number as positive data'
                             'in each training epoch')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='How many percentage of training data is splitted as validation set.')

    ## others
    parser.add_argument('--classifier', type=str, default='RF')
    parser.add_argument('--add_missing', type=int, default=0,
                        help='Add missingness indicator as feature')

    parser.add_argument('--mimic_cls', type=str, default='MIMIC_window')

    parser.add_argument('--num_features', type=int, default=40)
    parser.add_argument('--n_covs', type=int, default=0)
    parser.add_argument('--rnn_imputation', type=str, default='mean_imputation')

    parser.add_argument('--identifier', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--class_weight', type=str, default=None)

    args = parser.parse_args()
    np.random.seed(args.seed)

    return args


if __name__ == '__main__':
    args = parse_args()
    run()




