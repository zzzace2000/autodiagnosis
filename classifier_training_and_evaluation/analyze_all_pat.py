from arch.ManyToOneMGP_RNN import ManyToOneMGP_RNN, ManyToOneLMC_MGP_RNN
from database.MIMIC_window import MIMIC_window

import tensorflow as tf
from collections import Counter
import numpy as np


new_patient_database, database_hyperparams = MIMIC_window.load_database_by_category(
    data_dir='../data/my-mortality/',
    database_name='mingie_34features_6noise.pkl')

the_trained_dir = '../models/0916-34feats_6noise_negsampled-mimic-nh8-nl1-c1e-05-' \
                  'keeprate0.9_0.6_0.9-npred6-miss1-n_mc_15-MIMIC_window-' \
                  'mingie_34features_6noise-ManyToOneLMC_MGP_RNN/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    mgp_rnn = ManyToOneLMC_MGP_RNN.load_saved_mgp_rnn(sess, the_trained_dir)
    test_loader = new_patient_database.create_test_loader(batch_size=32)
    total_batches = next(test_loader)

    # train_loader, val_loader = new_patient_database.create_loaders(batch_size=32)
    # total_batches = next(train_loader)
    #
    # # In general, 0.15 +- 0.04 of probability if no measurement!!!
    # for obs_dict in train_loader:
    #     for j in range(len(obs_dict['Ys'])):
    #         obs_dict['Ys'][j] = []
    #         obs_dict['ind_kts'][j] = []
    #         obs_dict['ind_kfs'][j] = []
    #
    #     feed_dict = mgp_rnn.create_feed_dict_by_sparse_arr(**obs_dict)
    #     probs, = sess.run([mgp_rnn.probs], feed_dict)
    #
    #     probs = probs[:, :, 0, 1]  # Take only the class 1 probability
    #     avg_probs = np.average(probs, axis=-1)
    #     std_probs = np.std(probs, axis=-1)
    #     print('haha')

    label_aggregators = {}
    prob_avg_aggregators = {}
    prob_std_aggregators = {}
    for the_batch, obs_dict in enumerate(test_loader):
        feed_dict = mgp_rnn.create_feed_dict_by_sparse_arr(**obs_dict)
        O_pred, probs = sess.run([mgp_rnn.O_pred, mgp_rnn.probs], feed_dict)

        probs = probs[:, :, 0, 1]  # Take only the class 1 probability
        avg_probs = np.average(probs, axis=-1)
        std_probs = np.std(probs, axis=-1)

        if the_batch % 20 == 19:
            print('[%d/%d]' % (the_batch, total_batches))

        for step, total_step, avg_prob, std_prob, label in \
                zip(obs_dict['the_steps'], obs_dict['total_steps'],
                    avg_probs, std_probs, obs_dict['labels']):
            the_idx = total_step - 1 - step

            if the_idx not in prob_avg_aggregators:
                prob_avg_aggregators[the_idx] = []
                prob_std_aggregators[the_idx] = []
                label_aggregators[the_idx] = []

            prob_avg_aggregators[the_idx].append(avg_prob)
            prob_std_aggregators[the_idx].append(std_prob)
            label_aggregators[the_idx].append(label)

import pickle
pickle.dump((prob_avg_aggregators, prob_std_aggregators, label_aggregators),
            open('../notebooks/analyze_all_pats.pkl', 'wb'))
