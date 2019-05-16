import argparse
import json
import os

import numpy as np
import tensorflow as tf

from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN
from database.MIMIC_cache_exp import MIMIC_cache_discretized_joint_exp_random_order
from utils.general import Timer


def set_up_rl_agent(sess, dirname):
    """ Read in and load weight for rl agent """
    with open(os.path.join(dirname, 'hyperparams.log'), 'r') as file:
        hyperparams = json.load(file)
        policy_hyparams = hyperparams['rl']

    with open(os.path.join(dirname, 'results.json'), 'r') as file:
        training_results = json.load(file)

    # set up trained policy ----------------
    with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
        dqn = SequencialDuelingDQN(**policy_hyparams)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dqn'))

    saver.restore(sess, args.policy_dir + 'model-%d' % training_results['best_model_idx'])

    return dqn


def choose_topK_actions(sess, dqn, exp_loader, topK, debug=False):
    """ Choose top K actions given a dqn agent"""
    best_action_result = []
    num_actions_till_time_pass_result = []

    for batch_idx, exp in enumerate(exp_loader):
        batch_size, action_dim = exp['actions'].shape

        # Record best actions and corresponding len
        best_action_arrs = -1 * np.ones((batch_size, topK))
        num_actions_till_time_pass = topK * np.ones((batch_size,))

        # Start with empty history
        cur_history = np.zeros((batch_size, action_dim - 1))
        kept = np.ones((batch_size,), dtype=bool)

        for k in range(topK):
            if not np.any(kept):  # All records are time pass
                break

            s = np.concatenate((exp['next_state'], cur_history), axis=1)  # why next state?

            best_action = sess.run([dqn.best_actions], feed_dict={dqn.s: s})[0]
            best_action_arrs[kept, k] = best_action[kept]

            # Update num_actions! Last are kept but now becomes time pass
            num_actions_till_time_pass[kept & (best_action == action_dim - 1)] = (k + 1)

            # If action is time shift, flag it as non-kept
            kept = (kept & (best_action != action_dim - 1))

            cur_history[kept, best_action[kept]] = 1

        best_action_result.append(best_action_arrs)
        num_actions_till_time_pass_result.append(num_actions_till_time_pass)

        if debug and batch_idx == 20:
            break

    best_action_result = np.concatenate(best_action_result, axis=0)
    num_actions_till_time_pass_result = np.concatenate(num_actions_till_time_pass_result, axis=0)

    return best_action_result, num_actions_till_time_pass_result


def parse_args(rand=False):
    parser = argparse.ArgumentParser(description='Regression based value estimator')

    # Experience --------------------------------
    parser.add_argument('--cache_dir', type=str,
                        default='../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/')
    parser.add_argument('--policy_dir', type=str,
                        default='../models/dqn_mimic-1221_random_order_search-g1-ac5.0e-04-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-3-1-256-lr-0.0001-reg-0.0001-0.5-s-256-5000-i-50-500-3-1/')
    parser.add_argument('--load_per_action_cost', type=bool, default=True)
    parser.add_argument('--identifier', type=str, default='debug_')
    parser.add_argument('--topk', type=int, default=5)

    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    args = parse_args()

    # set up cached experience
    mimic_exp = MIMIC_cache_discretized_joint_exp_random_order(cache_dir=args.cache_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dqn = set_up_rl_agent(sess, args.policy_dir)

        test_loader = mimic_exp.gen_experience(sess=sess, filename='test_per_time', batch_size=1000, shuffle=False)

        # Action corresponding to the regression
        with Timer('Estimating the reward'):
            best_action_result, num_actions_till_time_pass_result = choose_topK_actions(sess=sess, dqn=dqn,
                                                                                        exp_loader=test_loader,
                                                                                        topK=args.topk, debug=True)

    print('done')
