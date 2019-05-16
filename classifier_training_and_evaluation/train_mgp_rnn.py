import argparse
import datetime
import json
import os
import time
import traceback
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from database.MIMIC_window import MIMIC_window
from utils.general import log
from utils.model_running import EarlyStopper
from arch.RNN import ManyToOneRNN, ManyToOneMGP_RNN
from arch.RNN_Survival import ManyToOneRNN_Survival


def construct_msg(epoch, total_epochs, metrics, spend_time=None):
    msg = "Epoch [{:d}/{:d}] ".format(epoch, total_epochs)
    msg += "loss_fit: {:.1e} loss_reg: {:.1e}".format(
        metrics['loss'], metrics['reg_loss'])

    if 'acc' in metrics:
        msg += " avg_acc: {:.3f} true_positive: {:.3f}".format(
            metrics['acc'], metrics['true_positive'])
    if 'auroc' in metrics:
        msg += " auroc: {:.3f} aupr: {:.3f}".format(metrics['auroc'], metrics['aupr'])
    if 'r2' in metrics:
        msg += ' r2: {:3f}'.format(metrics['r2'])

    if spend_time is not None:
        msg += '\nSpend %.0f seconds' % (spend_time)

    return msg


def train_iter(sess, mgp_rnn, patient_database, training_iters, batch_size,
               epoch_print, saver,
               early_stopper, logfile, model_path):
    for epoch in range(training_iters):
        start_time = time.time()
        log(logfile, "Starting epoch " + "{:d}".format(epoch))

        train_loader, val_loader = patient_database.create_loaders(batch_size)

        train_metrics = mgp_rnn.eval_train_loader(sess, train_loader, batch_print=50,
                                                  rnn_input_keep_prob=args.rnn_input_keep_prob,
                                                  rnn_output_keep_prob=args.rnn_output_keep_prob,
                                                  rnn_state_keep_prob=args.rnn_state_keep_prob)
        val_metrics = mgp_rnn.eval_test_loader(sess, val_loader)

        early_stopper.update(val_metrics)

        if epoch % epoch_print == 0:
            log(logfile, "# Train Set ----------------------------\n{}\n"
                         "# Val Set ----------------------------\n{}".format(
                construct_msg(epoch, training_iters, train_metrics),
                construct_msg(epoch, training_iters, val_metrics, time.time() - start_time),
            ) + ' best {}: {:.3f} in epoch {}'.format(early_stopper.metric,
                                                      early_stopper.get_best_metric(),
                                                      early_stopper.get_best_epoch()))

            if args.eval_test:
                test_loader = patient_database.create_test_loader(batch_size)
                test_metrics = mgp_rnn.eval_test_loader(sess, test_loader)
                log(logfile, "# Test Set ----------------------------\n{}".format(
                    construct_msg(epoch, training_iters, test_metrics)))

        saver.save(sess, '%s/%s' % (model_path, 'model'), global_step=epoch)

        # Decrease learning rate if not improving for 2 epochs
        if early_stopper.is_high_lr(epoch):
            mgp_rnn.decrease_lr(sess)

        # Early stopping for 3 epochs
        if early_stopper.is_ready(epoch):
            break


def train_MGP_RNN(sess, mgp_rnn, patient_database, batch_size, training_iters, logfile,
                  model_path, early_stopper, epoch_print=1):

    # Since early stopping do around 3 epochs.
    saver = tf.train.Saver(max_to_keep=(training_iters + 1))

    try:
        train_iter(sess, mgp_rnn, patient_database, training_iters, batch_size,
                   epoch_print, saver,
                   early_stopper, logfile, model_path)
    except BaseException as e:
        # Handle all the wierd problems like Cholescky,
        # or even ctrl+c interrupt
        log(logfile, str(e))
        traceback.print_exc()
        if early_stopper.is_empty():
            log(logfile, 'Not even finish one epoch! Exit!')
            return None

    best_model_idx = early_stopper.get_best_epoch()
    num_epochs_run = early_stopper.get_num_epochs()

    saver.restore(sess, os.path.join(model_path, 'model-%d' % best_model_idx))
    log(logfile, 'Restore the best model from epoch %d' % best_model_idx)
    log(logfile, 'Start evluating in test set...')

    start_time = time.time()
    test_loader = patient_database.create_test_loader(batch_size)
    test_metrics = mgp_rnn.eval_test_loader(sess, test_loader)

    log(logfile, "# Test Set ----------------------------\n{}".format(
        construct_msg(epoch=0, total_epochs=0, metrics=test_metrics,
                      spend_time=time.time() - start_time)))

    result_dict = OrderedDict([('test_%s' % key, test_metrics[key]) for key in test_metrics])
    result_dict.update({
        'best_model_idx': best_model_idx,
        'num_epochs_run': num_epochs_run,
    })

    return result_dict


def _run(args, thedate):
    final_identifier = '%s-%s-mimic-nh%d-nl%d-c%.0e-keeprate%.1f_%.1f_%.1f-npred%d-miss%d-n_mc_%d-%s-%s-%s' % (
        thedate, args.identifier, args.num_hidden, args.num_layers,
        args.l2_penalty, args.rnn_input_keep_prob, args.rnn_output_keep_prob, args.rnn_state_keep_prob,
        args.num_hours_pred, args.add_missing, args.n_mc_smps,
        args.mimic_cls, args.database, args.rnn_cls)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Check if already optimized
    model_path = os.path.join(args.output_dir, final_identifier)
    if os.path.exists(os.path.join(model_path, 'model.log')) and not args.overwrite:
        print('Already optimized! Exit!')
        return {}

    hyperparam_dict = {}
    hyperparam_dict['database'] = dict(
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

    new_patient_database = MIMIC_window(**hyperparam_dict['database'])

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Set up classifier ------------------------------------------
    logfile = os.path.join(model_path, 'train.log')
    classifier_attributes = dict(learning_rate=args.lr, l2_penalty=args.l2_penalty,
                                 n_hidden=args.num_hidden,
                                 n_layers=args.num_layers,
                                 n_classes=args.n_classes,
                                 num_features=args.num_features, n_meds=0,
                                 n_covs=args.n_covs, n_mc_smps=args.n_mc_smps, logfile=logfile,
                                 add_missing=args.add_missing,
                                 init_lmc_lengths=args.init_lmc_lengths,
                                 rnn_imputation=args.rnn_imputation,
                                 rnn_cls=args.rnn_cls,
                                 num_hours_pred=args.num_hours_pred)

    hyperparam_dict['classifier'] = classifier_attributes

    tf.reset_default_graph()
    with tf.variable_scope('mgp_rnn', reuse=tf.AUTO_REUSE):
        the_rnn = eval(args.rnn_cls)
        mgp_rnn = the_rnn(**classifier_attributes)

    # Save all hyperparameters to the model directory
    with open(os.path.join(model_path, 'hyperparams.log'), 'w') as op:
        json.dump(hyperparam_dict, op)

    early_stopper = EarlyStopper(metric=args.metric, is_max=True,
                                 num_early_stopping_epochs=args.lookahead)
    # Start training --------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        result_dict = train_MGP_RNN(
            sess=sess, mgp_rnn=mgp_rnn, patient_database=new_patient_database,
            batch_size=args.batch_size, training_iters=args.training_iters,
            logfile=logfile, model_path=model_path, early_stopper=early_stopper)

        if result_dict is None:
            return

    # Store config and perf ------------------------------------------
    with open(os.path.join(model_path, 'model_performances.json'), 'w') as op:
        json.dump(result_dict, op)

    if not os.path.exists('./performance/'):
        os.mkdir('./performance/')

    perf_file_name = './performance/{}_{}_{}.csv'.format(
        args.identifier, args.rnn_cls, args.mimic_cls)
    is_exist = os.path.exists(perf_file_name)
    with open(perf_file_name, 'a+') as op:
        if not is_exist:
            params_header = ['num_hours_pred', 'add_missing', 'identifier', 'num_hidden',
                             'num_layers', 'l2_penalty', 'n_mc_smps',
                             'rnn_input_keep_prob', 'rnn_output_keep_prob', 'rnn_state_keep_prob']
            print(','.join(params_header + list(result_dict.keys())), file=op)

        params_value = [str(args.num_hours_pred), str(args.add_missing), final_identifier,
                        str(args.num_hidden), str(args.num_layers), str(args.l2_penalty),
                        str(args.n_mc_smps),
                        str(args.rnn_input_keep_prob), str(args.rnn_output_keep_prob),
                        str(args.rnn_state_keep_prob)]
        print(','.join(params_value + [str(v) for v in result_dict.values()]), file=op)

    log(logfile, "done")


def run(args, thedate):
    start_time = time.time()
    _run(args, thedate)
    print('Finish evaluating. Spend {:.0f} seconds'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## Database
    parser.add_argument('--database_dir', type=str, default='../data/my-mortality/',
                        help='Where the dataset folder is')
    parser.add_argument('--database', type=str,
                        default='mingjie_39features_38covs',
                        help='The pickle file that must ends with .pkl. '
                             'E.g. Default file is in '
                             '../data/my-mortality/mingjie_39features_38covs.pkl')
    parser.add_argument('--num_hours_pred', type=float, default=24,
                        help='How many hours far to forecast if patient dies')
    # The data loader returns the corresponding time-series data:
    # We take the time interval between (-include_before_death, -before_end]
    # (value 0 is the end of the trajectory). Then start with time "-before_end",
    # we select the time point backward in time with interleaving of "data_interval" hours
    # until it's smaller than "-include_before_death".
    # E.g. for the last 24 hours of patient, we want to evaluate 3 hours interval each,
    # leading to total 9 points.
    # We can set <1> before_end as 0. <2> include_before_death 24.01
    # <3> data_interval as 3
    parser.add_argument('--before_end', type=float, default=0.)
    parser.add_argument('--include_before_death', type=float, default=24.01)
    parser.add_argument('--data_interval', type=float, default=3,
                        help='The evaluation interval')
    # We need these 2 to generate the RNN input time. Actually it should be put in the RNN classes...
    # It starts with the prediction time and go backward in time to choose "num_X_pred" points with
    # interval as "X_interval". These construct Xs which is the time for RNN input
    # E.g. We want to take the 24 points as RNN input with 1 hour seperate each. Set:
    #      X_interval as 1 and num_X_pred as 24
    parser.add_argument('--X_interval', type=float, default=1,
                        help='This specifies the time interval of RNN. Default is 1 hour.')
    parser.add_argument('--num_X_pred', type=float, default=24,
                        help='This specifies the number of point feeding into RNN.')
    # Filter data settings: make sure the input to RNN has at least "num_hours_warmup" hours
    # and "min_measurements_in_warmup" measurements
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

    ## RNN settings
    parser.add_argument('--num_features', type=int, default=39)
    parser.add_argument('--n_covs', type=int, default=38)
    parser.add_argument('--l2_penalty', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_hidden', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--n_mc_smps', type=int, default=1)
    parser.add_argument('--rnn_input_keep_prob', type=float, default=0.7)
    parser.add_argument('--rnn_output_keep_prob', type=float, default=0.7)
    parser.add_argument('--rnn_state_keep_prob', type=float, default=0.5)
    parser.add_argument('--rnn_imputation', type=str, default='mean_imputation',
                        help="choice from ['mean_imputation', 'forward_imputation']")
    parser.add_argument('--metric', type=str, default='auroc',
                        help="choose from [auroc, aupr]. Used for early stopping.")
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--add_missing', action='store_true', default=True)
    rnn_classes = [Cls.__name__ for Cls in [ManyToOneRNN, ManyToOneMGP_RNN,
                                            ManyToOneRNN_Survival]]
    parser.add_argument('--rnn_cls', type=str, default='ManyToOneRNN',
                        choices=rnn_classes)
    parser.add_argument('--init_lmc_lengths', nargs='+', type=float,
                        default=None)

    ## General Training
    parser.add_argument('--identifier', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='../models/')
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--training_iters', type=int, default=30)
    parser.add_argument('--eval_test', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lookahead', type=int, default=3)
    parser.add_argument('--num_random_run', type=int, default=0)

    args = parser.parse_args()

    # Now only supports this class
    args.mimic_cls = 'MIMIC_window'

    # Do random search or not
    args.random_mode = (args.num_random_run > 0)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    return args


if __name__ == "__main__":
    args = parse_args()
    thedate = datetime.datetime.now().strftime('%m%d')

    if not args.random_mode:
        run(args, thedate)
        exit()

    for i in range(args.num_random_run):
        print('The %d run:' % i)

        args.num_hidden = int(np.random.choice([32, 64, 128]))
        args.num_layers = int(np.random.choice([1, 2]))
        args.l2_penalty = np.random.choice([1e-7, 1e-6])

        args.rnn_input_keep_prob = np.random.choice([0.7, 0.9])
        args.rnn_output_keep_prob = np.random.choice([0.7, 0.9])
        args.rnn_state_keep_prob = np.random.choice([0.5, 0.7])

        # args.n_mc_smps = int(np.random.choice([1, 5, 20, 50]))

        run(args, thedate)
