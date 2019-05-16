'''
Given state and action, estimate gain er_action
'''

import argparse
import datetime
import os
import random
import json
import shutil

import numpy as np
import tensorflow as tf

from utils.general import get_logger2, output_csv
from utils.model_running import collect_step_results
from arch.RewardEstimator import StateToProbGainPerStepEstimator, StateToProbGainKTailsPerStepEstimator, StateToStateDiffPerStepEstimator, StateToStateDiffPerTimeEstimator, StateToProbGainPerTimeEstimator, StateToObsDiffPerTimeEstimator, ObsToObsDiffPerTimeEstimator, ObsToObsPerTimeEstimator,ObsToProbGainPerTimeEstimator, StateToObsPerTimeEstimator, StateDiffEstimatorMixin, StateToStatePerTimeEstimator
from arch.RNN import ManyToOneRNN


def is_training_complete(results):
    return results['num_epochs_run'] >= (args.max_training_iters - 1) or \
           results['num_epochs_run'] - results['best_model_idx'] >= args.early_stopping_epochs


def is_already_trained(model_path):
    has_file = os.path.isfile(os.path.join(model_path, 'results.json'))
    if not has_file:
        return None

    results = json.load(open(os.path.join(model_path, 'results.json'), "r"))

    if is_training_complete(results):
        return results
    return None


def reporter(epoch, train_results, val_results, test_results, test, log):
    s = ' loss {:.3}, reg: {:.3}, pea: {:.3}, r2 {:.3} '
    get_values = lambda d: [d['loss'], d['reg_loss'], d['pea'], d['r_square']]

    if test:
        log('test =>' + s.format(*get_values(test_results)))
    else:
        log(
            'Epoch {}:'.format(epoch) +
            'train =>' + s.format(*get_values(train_results)) +
            'val =>' + s.format(*get_values(val_results)))


def run_model(sess, model, model_path, log):
    def train_model(saver):

        log('# Training model ----------------')

        results = {'best_model_idx': 0, 'num_epochs_run': 0}
        train_writer = tf.summary.FileWriter(os.path.join(model_path, 'train'),
                                             sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(model_path, 'val'))

        for epoch in range(args.max_training_iters):
            log(f'# Train epoch {epoch} ----------------')
            train_loader = model.get_data_loader(mode='train', sess=sess, batch_size=args.train_batch_size)
            train_results = model.fit_loader(sess=sess, exp_loader=train_loader, process_exp_func=model.process_exp_func,
                                             is_training=True, writer=train_writer)

            log(f'# Val epoch {epoch} ----------------')
            val_loader = model.get_data_loader(mode='val', sess=sess, batch_size=args.test_batch_size)
            val_results = model.fit_loader(sess=sess, exp_loader=val_loader, process_exp_func=model.process_exp_func,
                                           is_training=False, writer=val_writer)
            reporter(epoch, train_results, val_results, test_results=None, test=False, log=log)

            saver.save(sess, os.path.join(model_path, 'model'), global_step=results['num_epochs_run'])

            collect_step_results(results, train_results, 'train')
            collect_step_results(results, val_results, 'val')

            results['best_model_idx'] = int(np.argmin(results['val_loss']))

            if results['num_epochs_run'] - results['best_model_idx'] > 1:
                model.decrease_lr(sess)

            with open(os.path.join(model_path, 'results.json'), 'w') as f:
                json.dump(results, f)

            if is_training_complete(results):
                log(f'Best epoch: {results["best_model_idx"]}, '
                    f'loss: {results["val_loss"][results["best_model_idx"]]}')
                break

            results['num_epochs_run'] += 1

        return results

    def test_model(saver, results):
        log('# Tesing model ----------------')

        best_model_idx = results['best_model_idx']

        saver.restore(sess, os.path.join(model_path, 'model-%d' % best_model_idx))
        log('Restore the best model from epoch %d' % best_model_idx)

        test_loader = model.get_data_loader(mode='test', sess=sess, batch_size=args.test_batch_size)
        test_results = model.fit_loader(sess=sess, exp_loader=test_loader,
                                        process_exp_func=model.process_exp_func,
                                        is_training=False)

        # Now only supports State diff decorator
        if isinstance(model, StateDiffEstimatorMixin):
            assert args.rnn_dir is not None

            # Put rnn inside the model
            rnn = ManyToOneRNN.load_saved_mgp_rnn(sess, model_dir=args.rnn_dir)
            model.rnn = rnn

            test_loader = model.get_data_loader(mode='test', sess=sess, batch_size=args.test_batch_size)

            all_outputs, all_targets = [], []
            for exp in test_loader:
                pred_prob_gain = model.get_predicted_prob_gain(sess, exp)

                all_outputs.append(pred_prob_gain)
                all_targets.append(exp['prob_gain'])

            all_outputs, all_targets = np.hstack(all_outputs), np.hstack(all_targets)
            test_results['prob_gain_pea'] = model._calculate_pearson_correlation(
                all_targets=all_targets, all_outputs=all_outputs)
            test_results['prob_gain_spearman_corr'] = model._calculate_spearman_correlation(
                all_targets=all_targets, all_outputs=all_outputs)
            test_results['prob_gain_r_square'] = model._calculate_r_square(
                all_targets=all_targets, all_outputs=all_outputs)

        reporter(epoch=None, train_results=None, val_results=None, test_results=test_results,
                 test=True, log=log)

        # Output
        the_dir = './estimation/'
        if not os.path.exists(the_dir):
            os.mkdir(the_dir)

        test_results.update({
            'identifier': args.identifier,
            'model_type': args.model_type,
            'lr': args.lr,
            'keep_prob': args.keep_prob,
            'reg_constant': args.reg_constant,
            'num_hidden_units': args.num_hidden_units,
            'num_hidden_layers': args.num_hidden_layers,
            'normalized_state': args.normalized_state,
            'num_shared_layers': args.num_shared_layers,
            'num_sep_layers': args.num_sep_layers,
            'all_identifier': args.all_identifier,
        })

        the_path = os.path.join(the_dir, '%s_%s.csv' % (args.identifier, args.model_type))
        output_csv(the_path, test_results)

    results = is_already_trained(model_path)
    saver = tf.train.Saver(max_to_keep=args.early_stopping_epochs + 1)

    if args.eval_mode == 0 and results is None:  # train is not complete
        results = train_model(saver)

    test_model(saver, results)


def parse_args():

    parser = argparse.ArgumentParser(description='Reward Estimator')

    # basic -----------------------
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--identifier', type=str, default='debug',
                        help='short identifier')

    # Mode
    parser.add_argument('--eval_mode', type=int, default=0)
    parser.add_argument('--eval_dir', type=str, default=None)
    parser.add_argument('--rnn_dir', type=str, default="../models/0117-24hours_39feats_38cov_negsampled_rnn-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred24-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN/")

    # classifier_training_and_evaluation ----------------
    # type 1 ~ 8
    all_cls_arr = [StateToProbGainPerStepEstimator, StateToProbGainKTailsPerStepEstimator,
                   StateToStateDiffPerStepEstimator, StateToStateDiffPerTimeEstimator,
                   StateToProbGainPerTimeEstimator, StateToObsDiffPerTimeEstimator,
                   ObsToObsDiffPerTimeEstimator, ObsToObsPerTimeEstimator,
                   ObsToProbGainPerTimeEstimator, StateToObsPerTimeEstimator,
                   StateToStatePerTimeEstimator]
    parser.add_argument('--model_type', type=str,
                        default='StateToProbGainPerTimeEstimator',
                        help='Choose from ' + str(all_cls_arr))

    # Experience --------------------------------
    parser.add_argument('--mimic_cache_cls', type=str, default=None,
                        help='If None, just choose MIMIC_cache_discretized_exp_env_v3')
    parser.add_argument('--cache_dir', type=str,
                        default='../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/')

    parser.add_argument('--normalized_state', type=int, default=1)

    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--num_shared_layers', type=int, default=1)
    parser.add_argument('--num_sep_layers', type=int, default=1)
    parser.add_argument('--num_hidden_units', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--reg_constant', type=float, default=1e-2)
    parser.add_argument('--keep_prob', type=float, default=0.8)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=5000)
    parser.add_argument('--max_training_iters', type=int, default=100)
    parser.add_argument('--early_stopping_epochs', type=int, default=5)

    parser.add_argument('--rand', type=int, default=0)
    parser.add_argument('--num_runs', type=int, default=20)
    parser.add_argument('--overwrite', type=int, default=0)

    args = parser.parse_args()
    print(args)

    if args.rand:
        args.overwrite = 0

    # Only starting with State could use this normalized flag
    if args.normalized_state and not args.model_type.startswith('State'):
        args.normalized_state = 0

    args.normalized_file = None
    if args.normalized_state:
        args.normalized_file = os.path.join(args.cache_dir, 'state_mean_and_std.csv')
        assert os.path.isfile(args.normalized_file)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    # Should turn this on. If we want diff. rand run, remember to change seed
    random.seed(args.seed)

    return args


def main(args):
    model_path = '../models/%s/' % args.all_identifier
    if args.overwrite or (os.path.exists(model_path) and not is_already_trained(model_path)):
        # do not overwrite random mode results
        print('Remove the model dir since overwrite or is not already trained')
        shutil.rmtree(model_path, ignore_errors=True)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    hyperparams = dict(cache_dir=args.cache_dir, model_type=args.model_type,
                       lr=args.lr, keep_prob=args.keep_prob,
                       reg_constant=args.reg_constant,
                       num_shared_layers=args.num_shared_layers,
                       num_sep_layers=args.num_sep_layers,
                       num_hidden_layers=args.num_hidden_layers,
                       num_hidden_units=args.num_hidden_units,
                       normalized_file=args.normalized_file,
                       mimic_cache_cls=args.mimic_cache_cls)
    if args.eval_mode == 0: # Only do it in non-eval mode
        with open(os.path.join(model_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f)

    log = get_logger2(os.path.join(model_path, 'log.txt'))
    log(model_path)

    tf.reset_default_graph()

    model_cls = eval(args.model_type)
    with tf.variable_scope(model_cls.variable_scope):
        model = model_cls(log=log, **hyperparams)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.variables_initializer(tf.global_variables(model.variable_scope)))

        run_model(sess=sess, model=model, model_path=model_path, log=log)


def hyperparameter_search(args):
    # Randomize hyperparamete ---------------------
    if args.rand:
        args.lr = random.choice([1e-2, 1e-3, 1e-4])
        args.keep_prob = random.choice([1.0, 0.9, 0.7, 0.5])
        args.reg_constant = random.choice([1e-2, 1e-3, 1e-4, 1e-5, 0.])
        args.num_hidden_units = random.choice([8, 16, 32, 64, 128, 256])

        args.num_hidden_layers = random.choice([1, 2, 3, 4, 5])
        args.num_shared_layers = random.choice([1, 2, 3, 4, 5])
        args.num_sep_layers = random.choice([1, 2, 3, 4, 5])

    thedate = datetime.datetime.now().strftime('%m%d')
    args.all_identifier = '{}-{}-{}-hl{}-hu{}-lr{}-reg{}-kp{}-n{}'.format(
        args.identifier, args.model_type,
        thedate, args.num_hidden_layers, args.num_hidden_units,
        args.lr, args.reg_constant, args.keep_prob,
        args.normalized_state)

    return args


if __name__ == "__main__":
    args = parse_args()

    # Run evaluation instead of training!
    if args.eval_mode == 1:
        args.overwrite = 0
        if not is_already_trained(args.eval_dir):
            print('This directory does not finish training! Exit!', args.eval_dir)
            exit()

        # Recover all the args hyperparameters
        with open(os.path.join(args.eval_dir, 'hyperparameters.json')) as f:
            hyperparams = json.load(f)

        args.all_identifier = os.path.basename(os.path.normpath(args.eval_dir))
        for k, v in hyperparams.items():
            setattr(args, k, v)

        main(args)
        exit()

    for i in range(args.num_runs):
        args = hyperparameter_search(args=args)
        main(args)
