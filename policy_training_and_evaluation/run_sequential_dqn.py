'''
Train Agent policy using K tail DQN
'''

# Handle a wierd bug when plotting image on the none-display server
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')

import argparse
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.display.max_columns = 100

from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN, LastObsSequencialDuelingDQN
from arch.K_Tails_Dueling_DQN import KTailsDuelingDQN

from database.MIMIC_cache_exp import \
    MIMIC_cache_discretized_joint_exp_random_order, \
    MIMIC_cache_discretized_exp_env_v3, MIMIC_cache_discretized_joint_exp_independent_measurement, \
    MIMIC_cache_discretized_joint_exp_random_order_with_obs
from utils.general import get_logger2
import shutil


def get_per_action_cost(args, log):
    action_freq_fname = os.path.join(args.cache_dir, 'action_freq.csv')
    if args.load_per_action_cost and os.path.isfile(action_freq_fname):
        log('# Load measurement cost from {}'.format(action_freq_fname))
        df = pd.DataFrame.from_csv(action_freq_fname)
        # Make the maximum as 1
        per_action_cost = 1. / df.values[:, 0]
        per_action_cost = per_action_cost / per_action_cost.max()
        return -per_action_cost
    else:
        log('# Generate generic measurement cost')

        # Note: no action the cost is 0, so no need to specify
        cost = - np.ones(args.action_dim)
        return cost


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


def run_model(args, sess, mimic_exp, mimic_per_time_env_exp,
              model, model_path, log):
    def report(all_results: dict):
        s = ' '.join(
            ['{}: {:5f}'.format(k, v[all_results['best_model_idx']])
             if isinstance(v, list) else '{}: {}'.format(k, v)
             for k, v in all_results.items()])
        log(s)

    def start_train():
        for epoch in range(args.max_training_iters):
            log(f'# Train epoch {epoch} ----------------')
            train_loader = mimic_exp.gen_train_experience(
                sess, batch_size=args.train_batch_size, shuffle=True)
            train_results = model.fit_loader(
                sess, train_loader, is_training=True,
                batch_print=400, writer=train_writer, debug=args.debug)

            log(f'# Val epoch {epoch} ----------------')
            val_loader = mimic_exp.gen_val_experience(sess, batch_size=args.test_batch_size, shuffle=False)
            val_results = model.fit_loader(
                sess, val_loader, get_best_action=False,
                is_training=False, batch_print=5, writer=val_writer, debug=args.debug)

            # Visualize validation action freq
            # val_per_time = mimic_per_time_env_exp.gen_experience(
            #     'val_per_time_env', sess, batch_size=args.test_batch_size, shuffle=False)
            # model.plot_tb_best_action_dist(sess, val_per_time, writer=val_writer, topK=5)

            # Log loss
            log('Epoch {}: train => loss {:.3}, reg: {:.3}, val => loss {:.3}, reg: {:.3}'.format(
                epoch, train_results['loss'], train_results['reg_loss'], val_results['loss'], val_results['reg_loss']))

            # log('Policy actions percentage:')
            # val_hist = np.histogram(np.concatenate(val_results['best_actions']), bins=np.arange(41))[0]
            # log(pd.DataFrame(val_hist[None, :] / val_results['total_num']))

            saver.save(sess, os.path.join(model_path, 'model'),
                       global_step=results['num_epochs_run'])

            for metric_name in ['loss', 'reg_loss']:
                results['train_' + metric_name].append(train_results[metric_name])
                results['val_' + metric_name].append(val_results[metric_name])

            results['best_model_idx'] = int(np.argmin(results['val_loss']))
            report(results)

            # Whenever one could not train well. Decrease lr!
            if results['num_epochs_run'] - results['best_model_idx'] > 0:
                model.decrease_lr(sess)

            json.dump(results, open(os.path.join(model_path, 'results.json'), 'w'))
            if is_training_complete(results):
                break

            results['num_epochs_run'] += 1

    def start_testing():
        log('# Load best model from local ----------------')
        saver.restore(sess, os.path.join(model_path, 'model-%d') % results['best_model_idx'])

        # Plot the best validation action freq!!
        def plot_best_action_freq(the_mode='val'):
            if not os.path.exists(os.path.join(args.cache_dir, '%s_per_time_env.tfrecords' % the_mode)):
                print('Skip plotting tb since no file exists')
                return
            # val_per_time = mimic_per_time_env_exp.gen_experience(
            #     '%s_per_time_env' % the_mode, sess, batch_size=args.test_batch_size, shuffle=False)
            # model.plot_tb_best_action_dist(
            #     sess, val_per_time, writer=val_writer, topK=5, image=True, identifier='%s_' % the_mode)

        log('Start plotting best action freq:')
        # plot_best_action_freq('train')
        plot_best_action_freq('val')
        plot_best_action_freq('test')

        # Test ---------------------------------
        log('# Testing model ----------------')
        test_loader = mimic_exp.gen_test_experience(sess, batch_size=args.test_batch_size)
        test_results = model.fit_loader(sess, test_loader, batch_print=None,
                                        is_training=False, get_best_action=False, debug=args.debug)

        # log('Test Policy actions percentage:')
        # test_hist = np.histogram(np.concatenate(test_results['best_actions']), bins=np.arange(41))[0]
        # log(pd.DataFrame(test_hist[None, :] / test_results['total_num']))
        # del test_results['best_actions']

        log('test => loss {:.3}, reg: {:.3}'.format(test_results['loss'], test_results['reg_loss']))
        json.dump(test_results, open(os.path.join(model_path, 'test_results.json'), 'w'))

    saver = tf.train.Saver(max_to_keep=args.early_stopping_epochs + 1)
    results = {'best_model_idx': 0, 'num_epochs_run': 0, 'train_loss': [], 'val_loss': [],
               'train_reg_loss': [], 'val_reg_loss': []}
    train_writer = tf.summary.FileWriter(os.path.join(args.tb_log_dir, args.my_identifier, args.identifier, 'train'),
                                         sess.graph) if args.tensorboard else None
    val_writer = tf.summary.FileWriter(os.path.join(args.tb_log_dir, args.my_identifier, args.identifier, 'val')) \
        if args.tensorboard else None

    is_trained_results = is_already_trained(model_path)
    if is_trained_results is None:  # train is not complete
        log('# Training model ----------------')
        start_train()
    else:
        log('Already trained!')

    start_testing()


def main():
    args.identifier = 'dqn_mimic-{}-g{}-ac{:.1e}-gamma{}-fold{}-only_pos{}-sd{}-ad{}-nn-{}-{}-{}-{}-lr-{}-reg-{}-{}-s-{}-{}-i-{}-{}-{}-{}'.format(
        args.my_identifier,
        args.gain_coef, args.action_cost_coef, args.gamma,
        args.pos_label_fold_coef, args.only_pos_reward,
        args.rl_state_dim, args.action_dim, args.memory_size,
        args.num_shared_all_layers, args.num_shared_dueling_layers, args.num_hidden_units,
        args.lr, args.reg_constant, args.keep_prob, args.train_batch_size, args.test_batch_size,
        args.max_training_iters, args.replace_target_batch, args.early_stopping_epochs, args.normalized_state
    )

    dqn_model_path = '../models/%s/' % args.identifier

    # Either replace flag is true or it has not finished training, remove the results
    if args.replace or (os.path.exists(dqn_model_path) and not is_already_trained(dqn_model_path)):
        shutil.rmtree(dqn_model_path, ignore_errors=True)

        tb_folder = os.path.join(args.tb_log_dir, args.my_identifier, args.identifier)
        if os.path.exists(tb_folder):
            shutil.rmtree(tb_folder, ignore_errors=True)

    # If already run, then do not override
    if not os.path.exists(dqn_model_path):
        os.makedirs(dqn_model_path)

    log = get_logger2(os.path.join(dqn_model_path, 'log.txt'))
    log(args)

    experience_attributes = dict(cache_dir=args.cache_dir)

    reward_attributes = dict(gain_coef=args.gain_coef, action_cost_coef=args.action_cost_coef, gamma=args.gamma,
                             pos_label_fold_coef=args.pos_label_fold_coef, only_pos_reward=args.only_pos_reward,
                             depend_on_labels=args.depend_on_labels)
    rl_attributes \
        = dict(variable_scope='dqn', state_dim=args.rl_state_dim,
               action_dim=args.action_dim,
               lr=args.lr, keep_prob=args.keep_prob, reg_constant=args.reg_constant,
               num_shared_all_layers=args.num_shared_all_layers,
               num_shared_dueling_layers=args.num_shared_dueling_layers,
               num_hidden_units=args.num_hidden_units,
               replace_target_batch=args.replace_target_batch, memory_size=args.memory_size,
               normalized_file=args.normalized_file, reward_attributes=reward_attributes)

    with open(os.path.join(dqn_model_path, 'hyperparams.log'), 'w') as op:
        hyperparam_dict = {'rl': rl_attributes, 'experience': experience_attributes, 'reward': reward_attributes,
                           'dqn_cls': args.dqn_cls}
        json.dump(hyperparam_dict, op)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # Set up expereince --------------------------
        mimic_cls = eval(args.cache_cls)
        mimic_exp = mimic_cls(**experience_attributes)
        # mimic_per_time_env_exp = MIMIC_cache_discretized_exp_env_v3(**experience_attributes)
        mimic_per_time_env_exp = None

        # Set up classifier_training_and_evaluation ------------------
        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            rl_attributes['log'] = log
            dqn_cls = eval(args.dqn_cls)
            dqn = dqn_cls(**rl_attributes)

        # Initialize params for classifier_training_and_evaluation ---------------
        sess.run(tf.variables_initializer(tf.global_variables(scope='dqn')))

        # Train DQN ----------------------
        run_model(args, sess, mimic_exp, mimic_per_time_env_exp, dqn, dqn_model_path, log)

    log("done")


def parse_args():
    parser = argparse.ArgumentParser(description='K Tails Dueling DQN')

    # basic -----------------------
    parser.add_argument('--rand', type=int, default=0)
    parser.add_argument('--num_random_run', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--my_identifier', type=str, default='1125_random_order_ac0_debug')
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--replace', type=int, default=1)

    # classifier_training_and_evaluation ----------------
    # Model
    dqn_choices = [SequencialDuelingDQN, KTailsDuelingDQN, LastObsSequencialDuelingDQN]
    parser.add_argument('--dqn_cls', type=str, default="LastObsSequencialDuelingDQN",
                        help="Choose from " + str(dqn_choices))
    parser.add_argument('--rl_state_dim', type=int, default=155)
    parser.add_argument('--action_dim', type=int, default=40)

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--memory_size', type=int, default=10000)

    # Training spec
    parser.add_argument('--normalized_state', type=int, default=0)
    parser.add_argument('--num_shared_all_layers', type=int, default=3)
    parser.add_argument('--num_shared_dueling_layers', type=int, default=1)
    parser.add_argument('--num_hidden_units', type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reg_constant', type=float, default=0.0001)
    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=5000)
    parser.add_argument('--replace_target_batch', type=int, default=500)
    parser.add_argument('--max_training_iters', type=int, default=50)
    parser.add_argument('--early_stopping_epochs', type=int, default=3)

    # reward design
    # parser.add_argument('--early_detecion_file', type=str, default='../notebooks/early_detection.csv')
    # parser.add_argument('--early_coef', type=float, default=0.05)
    parser.add_argument('--action_cost_coef', type=float, default=0.)
    parser.add_argument('--gain_coef', type=float, default=1)
    parser.add_argument('--pos_label_fold_coef', type=float, default=1)
    parser.add_argument('--only_pos_reward', type=int, default=0)
    parser.add_argument('--depend_on_labels', type=int, default=1)
    # state_mean_and_std.csv
    # Experience --------------------------------
    cache_cls_choices = [MIMIC_cache_discretized_joint_exp_random_order,
                         MIMIC_cache_discretized_joint_exp_independent_measurement,
                         MIMIC_cache_discretized_joint_exp_random_order_with_obs]
    parser.add_argument('--cache_cls', type=str,
                        default='MIMIC_cache_discretized_joint_exp_random_order_with_obs',
                        help=str(cache_cls_choices))
    parser.add_argument('--cache_dir', type=str,
                        default='../RL_exp_cache/0312-30mins-24hrs-20order-rnn-neg_sampled-with-obs/',
                        help='0117-30mins-24hrs-20order-rnn-neg_sampled')
    parser.add_argument('--load_per_action_cost', type=bool, default=True)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--tb_log_dir', type=str, default='../tb_logs/')
    args = parser.parse_args()

    # Set up hyperparameter ---------------------------
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    random.seed(args.seed)

    if args.debug:
        args.max_training_iters = 2
    if args.rand:
        args.replace = 0

    # Read the normalized
    args.normalized_file = None
    if args.normalized_state:
        args.normalized_file = os.path.join(args.cache_dir, 'state_mean_and_std.csv')
        assert os.path.isfile(args.normalized_file)

    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.rand:
        main()
        exit()

    # Randomize hyperparamete ---------------------
    for rand_idx in range(args.num_random_run):
        print('Starting random run %d' % rand_idx)
        # args.action_cost_coef = random.choice([1e-3, 1e-4, 1e-5, 0])
        # args.pos_label_fold_coef = random.choice([1])
        # args.only_pos_reward = random.choice([1])
        # args.normalized_state = random.choice([1])
        # args.gamma = random.choice([1, 0.8, 0.5, 0.2])
        args.lr = random.choice([1e-2, 1e-3, 1e-4, 1e-5])
        args.num_hidden_units = random.choice([32, 64, 128, 256, 512])
        args.num_shared_all_layers = random.choice([1, 2, 3, 4])
        args.reg_constant = random.choice([1e-2, 1e-3, 1e-4, 1e-5, 0.])
        args.keep_prob = random.choice([0.5, 0.7, 0.9])
        args.normalized_state = 1

        main()


