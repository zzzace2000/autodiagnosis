"""
Evaluate different policies in different environment for sequential setting
"""
import argparse
import pickle

import numpy as np
import tensorflow as tf

from arch.Sequencial_Dueling_DQN import SequencialDuelingDQN
from run_dqn_dummy import reward_function
from run_dqn_dummy import to_one_hot
from sim_observation_dummy_disease import ExperienceGeneratorSequential
from simulated_patient_database import DummyDisease, DummyClassifier, SimulatedPatientDatabaseDiscrete


def _get_random_action(history_action, done, num_to_choose, valid_action_inds):
    n_s, n_a = history_action.shape

    def per_s_helper(history_action_i):
        if np.sum(history_action_i) == min(num_to_choose, len(valid_action_inds)):
            return n_a

        usable_actions = [j for j in valid_action_inds if not history_action_i[j]]
        return np.random.choice(usable_actions)

    action_inds = np.array(list(map(per_s_helper, history_action)))

    return to_one_hot(action_inds, n_a + 1, done)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate policy')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward_params', type=float, default=[20, 5, 0])
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--num_patients', type=int, default=500)
    parser.add_argument('--early_detect_time', type=int, default=5)
    parser.add_argument('--dataset_setting_path', type=str,
                        default='../data/simulated_disease/1126_hyperparameters.pkl')
    parser.add_argument('--dqn_model_path', type=str,
                        default='../models/dqn-dummy1219-o50-a11-g0.0-0-m10000-nn1-3-16-lr0.001-0.0001-0.5-s128-15000-i1000-500-10-r(10, 5, 0)-d0.5-r0/')

    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    reward_func = lambda **kwargs: reward_function(reward_params=args.reward_params, **kwargs)

    dataset_setting = pickle.load(open(args.dataset_setting_path, 'rb'))

    classifier = DummyClassifier(num_useful_features=dataset_setting.num_useful_features,
                                 num_noisy_features=dataset_setting.num_noisy_features,
                                 obs_t_dim=dataset_setting.max_num_terminal_states,
                                 information_decay_rate=dataset_setting.information_decay_rate)

    disease = DummyDisease(feature_noise=dataset_setting.feature_noise,
                           max_num_terminal_states=dataset_setting.max_num_terminal_states,
                           num_useful_features=dataset_setting.num_useful_features,
                           num_noisy_features=dataset_setting.num_noisy_features,
                           num_datapoints_per_period=dataset_setting.num_datapoints_per_period,
                           period_length=dataset_setting.period_length,
                           min_periods=dataset_setting.min_periods, max_periods=dataset_setting.max_periods,
                           keep_rate=1.0)

    patient_database = SimulatedPatientDatabaseDiscrete(obs_type=disease, num_patients=args.num_patients)

    print('Rate of death: {}/{}'.format(np.sum(patient_database.all_labels), args.num_patients))

    exp_generator = ExperienceGeneratorSequential(patient_database=patient_database, classifier=classifier,
                                                  early_detect_time=args.early_detect_time, num_orders=1)


    def evaluate_random_policy(num_to_choose, informative=True):
        print(f'Num to choose: {num_to_choose}, informative: {informative}')
        get_random_informative_action = lambda delay_cur_state, history_action, done: \
            _get_random_action(history_action=history_action,
                               done=done,
                               num_to_choose=num_to_choose,
                               valid_action_inds=np.arange(disease.num_useful_features) \
                                   if informative else np.arange(disease.num_features))

        results = exp_generator.evaluate_batch(get_action=get_random_informative_action, reward_func=reward_func)

        results['gamma'] = args.gamma
        exp_generator.summarize(dic=results)


    def evaluate_dqn_policy(dqn_model_path):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        dqn_hyperparam = pickle.load(open(dqn_model_path + 'hyperparameters.pkl', 'rb'))
        dqn_results = pickle.load(open(dqn_model_path + 'results.pkl', 'rb'))

        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            dqn = SequencialDuelingDQN(variable_scope='dqn', state_dim=dqn_hyperparam.state_dim,
                                       action_dim=dqn_hyperparam.action_dim,
                                       lr=dqn_hyperparam.lr,
                                       keep_prob=dqn_hyperparam.keep_prob,
                                       reg_constant=dqn_hyperparam.reg_constant,
                                       num_shared_all_layers=dqn_hyperparam.num_shared_all_layers,
                                       num_shared_dueling_layers=dqn_hyperparam.num_shared_dueling_layers,
                                       num_hidden_units=dqn_hyperparam.num_hidden_units,
                                       replace_target_batch=dqn_hyperparam.replace_target_batch,
                                       memory_size=dqn_hyperparam.memory_size, log=None)

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dqn'))

        with tf.Session(config=config) as sess:
            saver.restore(sess, dqn_model_path + 'model-%d' % dqn_results['best_model_idx'])

            get_action_dqn = lambda delay_cur_state, history_action, done: \
                to_one_hot(inds=dqn.get_best_actions(sess=sess, s=delay_cur_state),
                           num_actions=disease.num_features + 1,
                           zero_inds=done)

            results = exp_generator.evaluate_batch(get_action=get_action_dqn, reward_func=reward_func)
            results['gamma'] = args.gamma

            exp_generator.summarize(dic=results)


    # for i in range(1, 6):
    #     evaluate_random_policy(num_to_choose=i, informative=True)
    #
    # for i in range(1, 11):
    #     evaluate_random_policy(num_to_choose=i, informative=False)
    evaluate_dqn_policy(dqn_model_path=args.dqn_model_path)
