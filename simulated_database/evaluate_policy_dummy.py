"""
Evaluate different policies in different environment
"""

import argparse
import pickle

import numpy as np
import tensorflow as tf

from arch.K_Tails_Dueling_DQN import KTailsDuelingDQN
from run_dqn_dummy import reward_function
from sim_observation_dummy_disease import ExperienceGeneratorMixed
from simulated_patient_database import DummyDisease, DummyClassifier, SimulatedPatientDatabaseDiscrete


def get_random_actions(s, action_dim):
    batch_size = len(s)
    return np.random.randint(0, 2, (batch_size, action_dim))


def get_random_informative_action(s, action_dim, useful_action_dim):
    batch_size = len(s)
    return np.concatenate((np.random.randint(0, 2, (batch_size, useful_action_dim)),
                           np.zeros((batch_size, action_dim - useful_action_dim))), axis=1)


def get_all_actions(s, action_dim):
    batch_size = len(s)
    return np.ones((batch_size, action_dim))


def get_all_informative_actions(s, action_dim, useful_action_dim):
    batch_size = len(s)
    return np.concatenate((np.ones((batch_size, useful_action_dim)),
                           np.zeros((batch_size, action_dim - useful_action_dim))), axis=1)

def get_single_informative_actions(s, action_dim, indices):
    batch_size = len(s)
    a = np.zeros((batch_size, action_dim))
    a[:, indices] = 1
    return a


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate policy')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward_params', type=float, default=(105,))

    parser.add_argument('--num_patients', type=int, default=500)

    parser.add_argument('--early_detect_time', type=int, default=5)

    parser.add_argument('--dataset_setting_path', type=str,
                        default='../data/simulated_disease/1015_hyperparameters.pkl')
    parser.add_argument('--data_keep_rate', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.65)

    args = parser.parse_args([])

    args.dqn_model_path_big_g = '../models/ktdqn-dummy1017-o30-a6-g0.99-m10000-nn1-2-64-lr1e-05-1e-05-0.5-' \
                                's64-5000-i20000-500-50-r(105,)-d0.9-r0/'
    return args


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    dataset_setting = pickle.load(open(args.dataset_setting_path, 'rb'))

    classifier = DummyClassifier(num_useful_features=dataset_setting.num_useful_features,
                                 num_noisy_features=dataset_setting.num_noisy_features,
                                 obs_t_dim=dataset_setting.max_num_terminal_states,
                                 score_param_1=dataset_setting.score_param_1)

    disease = DummyDisease(feature_noise=dataset_setting.feature_noise,
                           max_num_terminal_states=dataset_setting.max_num_terminal_states,
                           num_useful_features=dataset_setting.num_useful_features,
                           num_noisy_features=dataset_setting.num_noisy_features,
                           num_datapoints_per_period=dataset_setting.num_datapoints_per_period,
                           period_length=dataset_setting.period_length,
                           min_periods=dataset_setting.min_periods, max_periods=dataset_setting.max_periods,
                           keep_rate=args.data_keep_rate)

    patient_database = SimulatedPatientDatabaseDiscrete(obs_type=disease, num_patients=args.num_patients)

    print('Rate of death: {}/{}'.format(np.sum(patient_database.all_labels), args.num_patients))

    exp_generator = \
        ExperienceGeneratorMixed(patient_database=patient_database,
                                 classifier=classifier,
                                 early_detect_time=args.early_detect_time)

    reward_func = lambda **kwargs: reward_function(args.reward_params, **kwargs)


    def evaluate_dqn_policy(dqn_model_path, get_action_random, threshold, e_greedy=1.0):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        dqn_hyperparam = pickle.load(open(dqn_model_path + 'hyperparameters.pkl', 'rb'))
        dqn_results = pickle.load(open(dqn_model_path + 'results.pkl', 'rb'))

        tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
                dqn = \
                    KTailsDuelingDQN(variable_scope='dqn', state_dim=dqn_hyperparam.state_dim,
                                     action_dim=dqn_hyperparam.action_dim,
                                     gamma=dqn_hyperparam.gamma, lr=dqn_hyperparam.lr,
                                     keep_prob=dqn_hyperparam.keep_prob,
                                     reg_constant=dqn_hyperparam.reg_constant,
                                     num_shared_all_layers=dqn_hyperparam.num_shared_all_layers,
                                     num_shared_dueling_layers=dqn_hyperparam.num_shared_dueling_layers,
                                     num_hidden_units=dqn_hyperparam.num_hidden_units,
                                     replace_target_batch=dqn_hyperparam.replace_target_batch,
                                     memory_size=dqn_hyperparam.memory_size, log=None)

            saver = tf.train.Saver()
            saver.restore(sess, dqn_model_path + 'model-%d' % dqn_results['best_model_idx'])

            get_action = lambda s: dqn.get_best_actions(sess=sess, s=s) if np.random.rand() <= e_greedy \
                else get_action_random(s=s)

            cur_state, actions = exp_generator.evaluate_with_deterministic_statte(get_action=get_action)

            print(actions)

            avg_reward, avg_action, accuracy = exp_generator.evaluate(get_action=get_action, reward_func=reward_func,
                                                                      threshold=threshold)

            # print('Average reward: {}, Avg action: {}'.format(avg_reward, avg_action))
            return avg_reward, avg_action, accuracy


    def evaluate_easy_policy(get_action, threshold):
        avg_reward, avg_action, accuracy = exp_generator.evaluate(get_action=get_action,
                                                                  reward_func=reward_func,
                                                                  threshold=threshold)

        # print('Average reward: {}, Avg action: {}'.format(avg_reward, avg_action))
        return avg_reward, avg_action, accuracy


    get_action_random = lambda s: get_random_informative_action(s, action_dim=disease.num_features,
                                                                useful_action_dim=disease.num_useful_features)
    get_action_all = lambda s: get_all_informative_actions(s, action_dim=disease.num_features,
                                                           useful_action_dim=disease.num_useful_features)
    get_action_single0 = lambda s: get_single_informative_actions(s, action_dim=disease.num_features, indices=[0])
    get_action_single1 = lambda s: get_single_informative_actions(s, action_dim=disease.num_features, indices=[1])
    get_action_single2 = lambda s: get_single_informative_actions(s, action_dim=disease.num_features, indices=[2])
    get_action_single12 = lambda s: get_single_informative_actions(s, action_dim=disease.num_features, indices=[1, 2])
    get_action_single02 = lambda s: get_single_informative_actions(s, action_dim=disease.num_features, indices=[0, 2])

    res = evaluate_dqn_policy(dqn_model_path=args.dqn_model_path_big_g, get_action_random=get_action_random,
                              threshold=args.threshold)

    r, a, acc = res
    print('threshold', args.threshold, 'rew mean', np.mean(np.sum(r, axis=1), axis=0), 'act freq', np.mean(a), 'acc',
          np.mean(acc))
