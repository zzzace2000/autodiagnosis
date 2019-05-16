'''
Training agent policy using K tail dqn under simulated environment
'''
import argparse
import datetime
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN
from utils.general import get_logger, to_one_hot
from utils.model_running import is_training_complete, collect_step_results, is_already_trained, init_model_path


def run_model(sess, data_dict, model: 'SequencialDuelingDQN', model_path, train_batch_size, val_batch_size,
              max_training_iters, early_stopping_epochs, reward_func, _gamma, same_gamma):
    def get_gamma(a):
        if same_gamma:
            return np.ones(a.shape[0]) * _gamma
        else:
            non_time_pass = a[:, -1] != 1

            return np.ones(a.shape[0]) * non_time_pass + (1 - non_time_pass) * _gamma

    def get_data_loader(data_dict, batch_size, prioritized=False, shuffle=False):
        n = len(data_dict['cur_state'])

        if prioritized:
            idx_prio = np.any(data_dict['gain'], axis=1)
            n_prio = len(idx_prio)
            for k in data_dict:
                data_dict[k] = np.concatenate((data_dict[k][idx_prio], data_dict[k][~idx_prio][:n_prio]))

        if shuffle:
            idx_perm = np.random.permutation(n)
            for k in data_dict:
                data_dict[k] = data_dict[k][idx_perm]

        for i in range(0, n, batch_size):
            yield {k: data_dict[k][i:i + batch_size] for k in data_dict}

    def process_exp_func(d):
        """ Return RL s, a, r s_t, gamma """

        return np.concatenate([d['delay_cur_state'], d['cur_history_action']], axis=1), \
               d['action'], \
               reward_func(cur_action=d['action'], information_gain=d['gain'],
                           label=d['label']), \
               np.concatenate([d['delay_next_state'], d['next_history_action']], axis=1), \
               get_gamma(d['action'])

    def debug_code(sess, dqn, d):
        if d is None:
            return

        label = d['label'][:, 0]

        def _report_best_action_freq(cur_history_action):
            best_action = dqn.get_best_actions(sess=sess,
                                               s=np.concatenate([d['delay_cur_state'], cur_history_action], axis=1))
            best_action_1 = best_action[label]
            best_action_0 = best_action[1 - label]

            choice_0, freq_0 = np.unique(best_action_0, return_counts=True)
            choice_1, freq_1 = np.unique(best_action_1, return_counts=True)

            log(f'Action freq: \n'
                f'\t class 0 {choice_0}, {np.round(freq_0 / np.sum(freq_0), 3)}\n'
                f'\t class 1 {choice_1}, {np.round(freq_1 / np.sum(freq_1), 3)}')

            return best_action

        def _get_new_cur_history_action(cur_history_action, cur_action):
            num_actions = cur_history_action.shape[1] + 1
            cur_action_one_hot = to_one_hot(cur_action, num_actions=num_actions)
            return np.clip(cur_history_action + cur_action_one_hot[:, :-1], 0, 1)

        cur_history_action = d['cur_history_action']
        for i in range(3):
            best_action = _report_best_action_freq(cur_history_action)
            cur_history_action = _get_new_cur_history_action(cur_history_action=cur_history_action,
                                                             cur_action=best_action)

    def train_model():

        saver = tf.train.Saver(max_to_keep=early_stopping_epochs + 1)
        results = {'best_model_idx': 0, 'num_epochs_run': 0}
        train_writer = tf.summary.FileWriter(model_path + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(model_path + '/val')

        example_exp = pickle.load(open('../data/simulated_disease/1126_example_exp.pkl', 'rb'))

        for epoch in range(max_training_iters):
            debug_code(sess=sess, dqn=dqn, d=example_exp)

            log(f'# Train epoch {epoch} ----------------')
            train_loader = get_data_loader(data_dict=data_dict['train'], batch_size=train_batch_size,
                                           prioritized=False, shuffle=False)
            train_results = model.fit_loader(sess, train_loader, process_exp_func,
                                             is_training=True, batch_print=100, writer=train_writer)

            log(f'# Val epoch {epoch} ----------------')
            val_loader = get_data_loader(data_dict=data_dict['val'], batch_size=val_batch_size,
                                         prioritized=False, shuffle=False)
            val_results = model.fit_loader(sess, val_loader, process_exp_func,
                                           is_training=False, batch_print=5, writer=val_writer)

            log('Epoch {}: train => loss {:.3f}, reg: {:.3f}, val => loss {:.3f}, reg: {:.3f}'.format(
                epoch, train_results['loss'], train_results['reg_loss'], val_results['loss'], val_results['reg_loss']))

            saver.save(sess, os.path.join(model_path, 'model'),
                       global_step=results['num_epochs_run'])

            collect_step_results(results, train_results, 'train')
            collect_step_results(results, val_results, 'val')

            results['best_model_idx'] = int(np.argmin(results['val_' + 'loss']))

            if results['num_epochs_run'] - results['best_model_idx'] > 1:
                model.decrease_lr(sess)

            pickle.dump(results, open(os.path.join(model_path, 'results.pkl'), 'wb'))
            if is_training_complete(results, max_training_iters=max_training_iters,
                                    early_stopping_epochs=early_stopping_epochs):
                log(f'Best epoch: {results["best_model_idx"]}, loss: {results["val_loss"][results["best_model_idx"]]}')
                break

            results['num_epochs_run'] += 1
        return results

    results = is_already_trained(model_path, max_training_iters, early_stopping_epochs)

    if results is None:  # train is not complete
        log('# Training model ----------------')
        results = train_model()


def parse_args(rand):
    parser = argparse.ArgumentParser(description='Dueling DQN dummy')

    # basic -----------------------
    parser.add_argument('--seed', type=int, default=0)

    # Model
    parser.add_argument('--state_dim', type=int, default=50)
    parser.add_argument('--action_dim', type=int, default=11)

    parser.add_argument('--_gamma', type=float, default=0.999)
    parser.add_argument('--same_gamma', type=int, default=0)

    parser.add_argument('--memory_size', type=int, default=10000)

    # Training spec
    parser.add_argument('--num_shared_all_layers', type=int, default=1)
    parser.add_argument('--num_shared_dueling_layers', type=int, default=3)
    parser.add_argument('--num_hidden_units', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg_constant', type=float, default=1e-4)
    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=15000)
    parser.add_argument('--max_training_iters', type=int, default=1000)
    parser.add_argument('--replace_target_batch', type=int, default=500)
    parser.add_argument('--early_stopping_epochs', type=int, default=10)

    # reward design
    parser.add_argument('--reward_params', type=list, default=(20, 5, 0))

    # Experience --------------------------------
    parser.add_argument('--data_keep_rate', type=float, default=0.5)

    args = parser.parse_args([])
    # args.exp_fname = '../data/simulated_disease/1017_sr_simulated_6features_dummy_k{}.pkl'.format(args.data_keep_rate)
    args.exp_fname = f'../data/simulated_disease/1126_simulated_features_k{args.data_keep_rate}_sequential.pkl'

    # Randomize hyperparamete ---------------------

    if rand == 2:
        # args.num_shared_all_layers = random.choice([1, 2, 3, 4])
        # args.num_shared_dueling_layers = random.choice([1, 2, 3, 4])

        # args.num_hidden_units = random.choice([16, 32, 64, 128])

        # args.lr = random.choice([5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6])
        # args.reg_constant = random.choice([5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
        # args.keep_prob = random.choice([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        # args.train_batch_size = random.choice([32, 64, 128, 256, 512])
        args.same_gamma = random.choice([0, 1])
        args.reward_params[0] = random.choice(np.arange(10., 25., 1.0))
        args.reward_params[1] = random.choice(np.arange(3.0, 8., 0.5))

    thedate = datetime.datetime.now().strftime('%m%d')
    args.identifier = 'dqn-dummy{}-o{}-a{}-g{}-{}-m{}-nn{}-{}-{}-lr{}-{}-{}-s{}-{}-i{}-{}-{}-r{}-d{}-r{}'.format(
        thedate, args.state_dim, args.action_dim, args._gamma, args.same_gamma, args.memory_size,
        args.num_shared_all_layers, args.num_shared_dueling_layers, args.num_hidden_units,
        args.lr, args.reg_constant, args.keep_prob, args.train_batch_size, args.val_batch_size,
        args.max_training_iters, args.replace_target_batch, args.early_stopping_epochs,
        args.reward_params, args.data_keep_rate,
        int(rand)
    )

    return args


def reward_function(reward_params, cur_action, information_gain, label):
    """
    Reward formulation: linear combinatin of the gain and measurement cost and whether action is redundant
    different final label of the trajectory get to assign different weight on the gain
    """
    assert len(cur_action.shape) == 2 and len(information_gain.shape) == 2 and len(label.shape) == 2

    a, b, c = reward_params

    # set up action cost to 1
    non_time_passing = np.sum(cur_action[:, :-1], axis=-1, keepdims=True)
    action_cost = non_time_passing * 1

    # adjust information gain by label
    gain_label1 = (information_gain * a) * label
    gain_label0 = (information_gain * - b) * (1 - label)
    adjusted_information_gain = gain_label0 + gain_label1

    # adjust information gain of time passing operation
    if c:
        adjusted_information_gain *= non_time_passing

    return adjusted_information_gain - action_cost


if __name__ == "__main__":
    rand = 0
    args = parse_args(rand=rand)

    reward_func = lambda **kwargs: reward_function(reward_params=args.reward_params, **kwargs)

    model_path = init_model_path(rand, args.identifier)
    log = get_logger(model_path + 'log.txt')
    log(model_path)
    pickle.dump(args, open(model_path + 'hyperparameters.pkl', 'wb'))

    # Set up hyperparameter ---------------------------
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data_dict = pickle.load((open(args.exp_fname, 'rb')))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            dqn = SequencialDuelingDQN(variable_scope='dqn', state_dim=args.state_dim, action_dim=args.action_dim,
                                       lr=args.lr, keep_prob=args.keep_prob,
                                       reg_constant=args.reg_constant,
                                       num_shared_all_layers=args.num_shared_all_layers,
                                       num_shared_dueling_layers=args.num_shared_dueling_layers,
                                       num_hidden_units=args.num_hidden_units,
                                       replace_target_batch=args.replace_target_batch,
                                       memory_size=args.memory_size, log=log)

        # Initialize params for classifier_training_and_evaluation ---------------
        sess.run(tf.variables_initializer(tf.global_variables(scope='dqn')))

        run_model(sess=sess, data_dict=data_dict, model=dqn, model_path=model_path,
                  train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
                  max_training_iters=args.max_training_iters, early_stopping_epochs=args.early_stopping_epochs,
                  reward_func=reward_func, _gamma=args._gamma, same_gamma=args.same_gamma)

    print("done")
