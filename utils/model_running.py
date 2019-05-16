import os
import pickle

import numpy as np


# import pickle
# import os
# import tensorflow as tf
#
# from utils.general import log


def is_training_complete(results: dict, max_training_iters: int, early_stopping_epochs: int):
    ''' Determine whehter training is completed
    '''
    return results['num_epochs_run'] >= max_training_iters or \
           results['num_epochs_run'] - results['best_model_idx'] >= early_stopping_epochs


def handle_result(results: dict, batch_size: int, result:dict):
    '''
    Aggrregate minibatch result into results
    '''
    if result is None:
        return

    if results is None:
        results = {k: [[], []] for k in result}
    for k in result:
        if result[k] is not None:
            results[k][0].append(result[k])
            results[k][1].append(batch_size)
    return results


def handle_results(results):
    '''
    Summary minibatch results
    '''
    for k in results:
        if len(results[k][0]) > 0:
            nu = np.sum([results[k][0][i] * results[k][1][i] for i in range(len(results[k][0]))])
            de = np.sum(results[k][1])
            results[k] = nu / de
        else:
            results[k] = None
    return results


def collect_step_results(all_results: dict, step_results: dict, prefix: str):
    for k in step_results:
        if step_results[k] is not None:
            if prefix + '_' + k not in all_results:
                all_results[prefix + '_' + k] = []

            all_results[prefix + '_' + k].append(step_results[k])


def report(all_results: dict, log):
    s = ' '.join(['{}: {:5f}'.format(k, all_results[k][-1]) if type(all_results[k]) == type([]) else
                  '{}: {}'.format(k, all_results[k]) for k in all_results])
    log(s)
    s = ' '.join(
        ['{}: {:5f}'.format(k, all_results[k][all_results['best_model_idx']])
         if type(all_results[k]) == type([]) else '{}: {}'.format(k, all_results[k])
         for k in all_results])
    log(s)


def is_already_trained(model_path, max_training_iters, early_stopping_epochs):
    has_file = os.path.isfile(os.path.join(model_path, 'results.pkl'))
    if not has_file:
        return None

    results = pickle.load(open(os.path.join(model_path, 'results.pkl'), "rb"))
    if is_training_complete(results, max_training_iters=max_training_iters,
                            early_stopping_epochs=early_stopping_epochs):
        return results
    return None


def init_model_path(overwrite, identifier):
    model_path = '../models/%s/' % identifier

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    elif overwrite:
        for the_file in os.listdir(model_path):
            file_path = os.path.join(model_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        return
    return model_path


class EarlyStopper:
    def __init__(self, metric: str = 'auroc', is_max: bool = True,
                 num_early_stopping_epochs: int = 3,
                 num_lr_decay_epochs: int = 1):

        self.val_arr = []
        self.metric = metric
        self.num_early_stopping_epochs = num_early_stopping_epochs
        self.num_lr_decay_epochs = num_lr_decay_epochs

        if is_max:
            self.best_fn = np.max
            self.epoch_selection_fn = np.argmax
        else:
            self.best_fn = np.min
            self.epoch_selection_fn = np.argmin

    def is_empty(self):
        return len(self.val_arr) == 0

    def get_num_epochs(self):
        return len(self.val_arr)

    def get_best_metric(self):
        return self.best_fn(self.val_arr)

    def get_best_epoch(self):
        return int(self.epoch_selection_fn(self.val_arr))

    def update(self, val_metrics: dict):
        self.val_arr.append(val_metrics[self.metric])

    def is_ready(self, epoch):
        return epoch - self.get_best_epoch() > self.num_early_stopping_epochs

    def is_high_lr(self, epoch):
        return epoch - self.get_best_epoch() > self.num_lr_decay_epochs
