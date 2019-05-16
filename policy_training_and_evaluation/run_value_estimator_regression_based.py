import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN
from utils.general import to_one_hot, Timer, output_csv


def get_reward_attributes(dirname):
    with open(os.path.join(dirname, 'hyperparams.log'), 'r') as file:
        hparams = json.load(file)
        reward_attributes = hparams['reward']

    print(reward_attributes)

    return reward_attributes


def gen_rand_action(action_history, time_pass_freq_scale_factor=1):
    '''
    Generate one-hot random action but skip actions already performed in the history.
    :param time_pass_freq_scale_factor: Tune the frequency of time passing
    :param action_history: B x (D-1) array. D is the total action numbers
                           including time-pass action
    :return: B x D action array. Each row is a one-hot vector
    '''
    B, D = action_history.shape[0], action_history.shape[1] + 1

    rand_mat = np.random.rand(B, D)
    # Scale the time_pass val be smaller to get less prob. be selected
    rand_mat[:, -1] *= time_pass_freq_scale_factor
    # Set the history action's val as min
    tmp = np.concatenate((action_history, np.zeros((B, 1))), axis=1)
    rand_mat[tmp == 1] = -1
    # Take the maximum val as the action performed
    rand_a = rand_mat.argmax(axis=1)
    rand_a = to_one_hot(rand_a, num_actions=D)

    return rand_a


def _evalaute_exp(exp_loader, evaluate_exp_fn):
    '''
    Helpler fn to pass in the exp call back fn and return per-patient dataframe
    :param evaluate_exp_fn: args => exp, return => dict with key as name and val as np array
    :return pandas df. Each column is the key and each row is per patient summary (sum of all exps)
    '''
    result_dict = {}

    # Action corresponding to the regression
    with Timer('Estimating the reward'):
        for idx, exp in enumerate(exp_loader):
            evaluation_dict = evaluate_exp_fn(exp)

            for k, v in evaluation_dict.items():
                if k not in result_dict:
                    result_dict[k] = []
                result_dict[k].append(v)

        df = pd.DataFrame({k: np.concatenate(v, axis=0) for k, v in result_dict.items()})

        ## Get per-patient summary
        assert 'patient_inds' in df.columns and 'mortality' in df.columns
        group = df.groupby('patient_inds')

        # Since all the eval has mortality as the attribute, use it to instantiate
        df_aggregate = group['mortality'].apply(lambda x: x.iloc[0]).reset_index()

        # For all the other attributes, just sum over the number for each specific patient
        for k in df.columns:
            if k == 'patient_inds' or k == 'mortality':
                continue
            df_aggregate[k] = group[k].apply(lambda x: x.sum()).values

        return df_aggregate


def evaluate_rl_policy_per_step(args, sess, dqn, rew_estimator, reward_attributes, reward_func):
    '''
    Evaluates 1 agent, 1 phy and 1 random policy per timestep
    '''
    def tuple_fn(exp):
        # The delay_update_state should be used when in the intermediate steps
        time_pass = exp['actions'][:, -1:]
        cur_state = time_pass * exp['cur_state'] + (1 - time_pass) * exp['delay_update_state']
        s = np.concatenate((cur_state, exp['cur_history']), axis=-1)

        agent_a = to_one_hot(dqn.get_best_actions(sess, s=s), num_actions=dqn.action_dim)
        rand_a = gen_rand_action(action_history=exp['cur_history'])

        return cur_state, exp['actions'], agent_a, rand_a

    return _evaluate_rl_policy_helper(args, sess, rew_estimator, reward_attributes, reward_func,
                                      tuple_fn)


def evaluate_rl_policy_per_time(args, sess, dqn, rew_estimator, reward_attributes, reward_func):

    num_actions_performed, total_num_time = 0, 0

    def tuple_fn(exp):
        agent_a = dqn.get_best_sequential_actions(sess=sess, exp=exp)

        nonlocal num_actions_performed, total_num_time
        num_actions_performed += agent_a[:, :-1].sum()

        batch_size = agent_a.shape[0]
        total_num_time += batch_size

        # Generate random action for per-time experience. Use 6.6% action, same as phy.
        # The shape is only (B, 39) that do not contain the time-passing part
        rand_a = np.random.rand(batch_size, agent_a.shape[-1] - 1)
        rand_a[rand_a > 0.934] = 1
        rand_a[rand_a <= 0.934] = 0

        return exp['cur_state'], exp['cur_actions'], agent_a[:, :-1], rand_a

    _evaluate_rl_policy_helper(
        args, sess, rew_estimator, reward_attributes, reward_func, tuple_fn)

    # Count the % of actions
    print('Agent action freq per time: ', num_actions_performed / total_num_time)


def _evaluate_rl_policy_helper(args, sess, rew_estimator, reward_attributes, reward_func, tuple_fn):
    '''
    Evaluates 1 agent, 1 phy and 1 random policy
    '''
    test_result_path = os.path.join(args.policy_dir, 'test_results.json')
    if not os.path.isfile(test_result_path):
        exit('No test_results.json exist. This policy is not finished! Exit!')

    def evaluate_exp_fn(exp):
        '''
        Return a dict that has keys pointing to the result.
        '''
        result = {}

        # Since per-step and per-time exp are different
        cur_state, a, agent_a, rand_a = tuple_fn(exp)

        # Concatenate everything and run once
        agent_prob_gain = rew_estimator.get_predicted_prob_gain(sess, exp, agent_a=agent_a)
        phy_prob_gain = rew_estimator.get_predicted_prob_gain(sess, exp, agent_a=a)
        rand_prob_gain = rew_estimator.get_predicted_prob_gain(sess, exp, agent_a=rand_a)

        # Calculate reward
        agent_info_gain, agent_cost = reward_func(agent_prob_gain, exp['mortality'], agent_a, exp['labels'])
        phy_info_gain, phy_cost = reward_func(phy_prob_gain, exp['mortality'], a, exp['labels'])
        phy_real_info_gain, _ = reward_func(exp['prob_gain'], exp['mortality'], a, exp['labels'])
        rand_info_gain, rand_cost = reward_func(rand_prob_gain, exp['mortality'], rand_a, exp['labels'])

        decay = (reward_attributes['gamma'] ** exp['the_steps'])

        result['agent_info_gains'] = decay * agent_info_gain
        result['agent_action_costs'] = decay * agent_cost
        result['phy_info_gains'] = decay * phy_info_gain
        result['phy_action_costs'] = decay * phy_cost
        result['phy_real_info_gains'] = decay * phy_real_info_gain
        result['rand_info_gains'] = decay * rand_info_gain
        result['rand_action_costs'] = decay * rand_cost
        result['patient_inds'] = exp['patient_inds']
        result['mortality'] = exp['mortality']
        return result

    # Evaluate exp
    test_loader = rew_estimator.get_data_loader(mode='test', sess=sess, batch_size=1024)
    df_aggregate_test = _evalaute_exp(test_loader, evaluate_exp_fn)

    val_loader = rew_estimator.get_data_loader(mode='val', sess=sess, batch_size=1024)
    df_aggregate_val = _evalaute_exp(val_loader, evaluate_exp_fn)

    # Process val estimation summary
    def process_summary(df, mode='test'):
        # Store per-patient summary under different folder
        df.to_csv(os.path.join(args.policy_dir, '%s_%s_per_patient_summary.csv' % (args.identifier, mode)),
                  index=None)

        ret_dict = {}
        ret_dict['test_loss'] = json.load(open(test_result_path))['loss']
        ret_dict.update(dict(
            policy_dir=args.policy_dir,
            identifier=args.identifier,
            reward_estimator_dir=args.reward_estimator_dir,
            cache_dir=args.cache_dir,
            action_cost_coef=reward_attributes['action_cost_coef'],
            gain_coef=reward_attributes['gain_coef']
        ))

        for k in ['agent', 'phy', 'rand']:
            ret_dict['mean_%s_info_gains' % k] = df['%s_info_gains' % k].values.mean()
            ret_dict['mean_%s_action_costs' % k] = df['%s_action_costs' % k].values.mean()
        ret_dict['mean_phy_real_info_gains'] = df['phy_real_info_gains'].values.mean()

        return ret_dict

    test_dict = process_summary(df_aggregate_test, mode='test')
    val_dict = process_summary(df_aggregate_val, mode='val')

    for k in ['agent', 'phy', 'rand']:
        test_dict['val_mean_%s_info_gains' % k] = val_dict['mean_%s_info_gains' % k]
        test_dict['val_mean_%s_action_costs' % k] = val_dict['mean_%s_action_costs' % k]
    test_dict['val_mean_phy_real_info_gains'] = val_dict['mean_phy_real_info_gains']

    # Get val loss
    result_path = os.path.join(args.policy_dir, 'results.json')
    dqn_results = json.load(open(result_path))
    test_dict['val_loss'] = dqn_results['val_loss'][dqn_results['best_model_idx']]

    # Output ret_dict to the dir ./estimation/
    the_dir = './estimation/'
    if not os.path.exists(the_dir):
        os.mkdir(the_dir)

    the_path = os.path.join(the_dir, '%sval_regression_summary.csv' % args.identifier)
    output_csv(the_path, test_dict)


def evaluate_random_policy_per_time(args, sess, rew_estimator, reward_attributes, reward_func):
    def evaluate_exp_fn(exp):
        '''
        Return a dict that has keys pointing to the result.
        '''
        result = {}
        # Generate random action but follows the history constraint
        rand_a = np.random.rand(exp['cur_state'].shape[0], 39)
        rand_a[rand_a > args.time_pass_freq_scale_factor] = 1
        rand_a[rand_a <= args.time_pass_freq_scale_factor] = 0

        rand_prob_gain = rew_estimator.get_predicted_prob_gain(sess, exp, agent_a=rand_a)
        rand_info_gain, rand_cost = reward_func(rand_prob_gain, exp['mortality'], rand_a, exp['labels'])

        decay = (reward_attributes['gamma'] ** exp['the_steps'])
        result['rand_info_gains'] = decay * rand_info_gain
        result['rand_action_costs'] = decay * rand_cost
        result['patient_inds'] = exp['patient_inds']
        result['mortality'] = exp['mortality']
        return result

    for repeat in range(3):
        test_loader = rew_estimator.get_data_loader(mode='test', sess=sess, batch_size=1024)

        df_aggregate = _evalaute_exp(test_loader, evaluate_exp_fn)

        if not os.path.exists('estimation/%s/' % args.identifier):
            os.mkdir('estimation/%s/' % args.identifier)

        per_patient_path = 'estimation/%s/rand%d_tpf%f_per_patient_summary.csv' \
                           % (args.identifier, repeat, args.time_pass_freq_scale_factor)
        df_aggregate.to_csv(per_patient_path, index=None)

        info_gains, action_costs = df_aggregate['rand_info_gains'].values.mean(), \
                                   df_aggregate['rand_action_costs'].values.mean()
        print(info_gains, action_costs, args.time_pass_freq_scale_factor)

        result_dict = {
            'mean_rand_info_gains': info_gains,
            'mean_rand_action_costs': action_costs,
            'cache_dir': args.cache_dir,
            'reward_estimator_dir': args.reward_estimator_dir,
            'rnn_dir': args.rnn_dir,
            'per_patient_path': per_patient_path,
        }

        the_path = './estimation/%s_random_policy_evaluation.csv' % args.identifier
        output_csv(the_path, result_dict)


def evaluate_random_policy(args, sess, rew_estimator, reward_attributes, reward_func):
    def evaluate_exp_fn(exp):
        '''
        Return a dict that has keys pointing to the result.
        '''
        result = {}
        # Generate random action but follows the history constraint
        rand_a = gen_rand_action(action_history=exp['cur_history'],
                                 time_pass_freq_scale_factor=time_pass_freq_scale_factor)

        time_pass = exp['actions'][:, -1:]
        cur_state = time_pass * exp['cur_state'] + (1 - time_pass) * exp['delay_update_state']

        rand_i = np.concatenate([cur_state, exp['cur_history'], rand_a], axis=1)
        rand_prob_gain = rew_estimator.get_output_i(sess, input_i=rand_i)[:, 0]

        # Calculate reward
        rand_info_gain, rand_cost = reward_func(rand_prob_gain, exp['mortality'], rand_a, exp['labels'])

        decay = (reward_attributes['gamma'] ** exp['the_steps'])
        result['rand_info_gains'] = decay * rand_info_gain
        result['rand_action_costs'] = decay * rand_cost
        result['patient_inds'] = exp['patient_inds']
        result['mortality'] = exp['mortality']
        return result

    all_info_gains, all_action_costs = [], []
    for time_pass_freq_scale_factor in np.arange(0, 4, 0.2):
        test_loader = rew_estimator.get_data_loader(mode='test', sess=sess, batch_size=1024)
        df_aggregate = _evalaute_exp(test_loader, evaluate_exp_fn)
        info_gains, action_costs = df_aggregate['rand_info_gains'].values.mean(), \
                                   df_aggregate['rand_action_costs'].values.mean()
        all_info_gains.append(info_gains)
        all_action_costs.append(action_costs)

    # Save the results
    df = pd.DataFrame({
        'mean_rand_info_gains': np.array(all_info_gains),
        'mean_rand_action_costs': np.array(all_action_costs),
        'cache_dir': args.cache_dir,
        'policy_dir': args.policy_dir
    })

    print(df.head(5))
    df.to_csv('./estimation/%s_5random_policy_evaluation.csv' % args.identifier, index=None)


def parse_args():

    parser = argparse.ArgumentParser(description='Regression based value estimator')

    # Experience --------------------------------
    parser.add_argument('--policy_dir', type=str,
                        default='../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/')
    parser.add_argument('--rnn_dir', type=str,
                        default="../models/0117-24hours_39feats_38cov_negsampled_rnn-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred24-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN/")
    parser.add_argument('--reward_estimator_dir', type=str,
                        default='../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/')
    parser.add_argument('--mode', type=int, default=4)
    parser.add_argument('--identifier', type=str, default='debug2_')
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--time_pass_freq_scale_factor', type=float, default=0)

    args = parser.parse_args()
    # Just for old compatibility
    args.cache_dir = 'useless'
    return args


def main():
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    args = parse_args()

    # set up cached experience
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        dqn = SequencialDuelingDQN.load_saved_model(sess, the_dir=args.policy_dir)
        reward_attributes = dqn.reward_attributes

        def reward_func(prob_gain, mortality, actions, labels):
            """ Return raw information gain and raw action cost """
            # If mortality is 1, then encourage to increase probability.
            # If mortality is 0, be negative.
            if 'depend_on_labels' in reward_attributes and reward_attributes['depend_on_labels'] == 1:
                information_gain = prob_gain * (2 * labels - 1)
            else:
                information_gain = prob_gain * (2 * mortality - 1)

            if reward_attributes['only_pos_reward']:
                information_gain[information_gain < 0] = 0

            # action cost. Sum over all the other number of actions
            action_cost = actions[:, :40].sum(axis=1)

            return information_gain, action_cost

        from arch.RewardEstimator import load_model
        rew_estimator = load_model(sess, args.reward_estimator_dir, rnn_dir=args.rnn_dir)

        if args.mode == 1:  # per-step adjustment evaluation of one random policy, physician policy and one rl policy
            print('Start per-step estimation for folder %s' % args.policy_dir)

            evaluate_rl_policy_per_step(args=args, sess=sess, dqn=dqn, rew_estimator=rew_estimator,
                                        reward_attributes=reward_attributes, reward_func=reward_func)

        elif args.mode == 2:  # per-step adjustment evalution of five random policy
            print('Start 40 different random policy')
            evaluate_random_policy(args, sess, rew_estimator, reward_attributes, reward_func)

        elif args.mode == 3: # per-time adjustment evaluation
            print('Start per-time estimation for folder %s' % args.policy_dir)

            the_path = os.path.join('estimation', '%sval_regression_summary.csv' % args.identifier)
            if not args.overwrite and os.path.exists(the_path):
                all_policy_dirs = pd.read_csv(the_path)['policy_dir'].values
                if args.policy_dir in all_policy_dirs:
                    print('Already evaluate this policy ' + args.policy_dir)
                    exit()

            if not os.path.exists(os.path.join(args.policy_dir, 'test_results.json')):
                print('This directory has not finished! ' + args.policy_dir)
                exit()

            evaluate_rl_policy_per_time(args=args, sess=sess, dqn=dqn, rew_estimator=rew_estimator,
                                        reward_attributes=reward_attributes, reward_func=reward_func)

        elif args.mode == 4:  # per-step adjustment evalution of five random policy
            print('Start 40 different random policy')
            evaluate_random_policy_per_time(args, sess, rew_estimator, reward_attributes, reward_func)


if __name__ == '__main__':
    main()
