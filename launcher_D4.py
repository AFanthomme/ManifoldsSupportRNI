from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch


default_D4_pars = {
    'net_type': 'many_channels',
    'net_params': { 'train_ed': False,
                    'device_name': 'cuda',
                    'n': 1000,
                    'n_channels': 4,
                    'saturations': [-1e8, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'is_switch': False,
                    'activation_type': 'ReLU',
                    },
    'sampler_params': {
                    'n_channels': 4,
                    'epoch_length': 3,
                    'decays': [.995, 0.99, 0.992, 0.994],
                    'scales': [1., 1., 1., 1.],
                    'batch_size': 1024,
                    'is_switch': False,
                    },
    'train_params': {'train_ed': False,
                    'loss_name': 'batch',
                    'optimizer_name': 'sgd',
                    'lr': 1e-1,
                    'n_epochs': 5000,
                    'stop_loss': 1e-9,
                    },
    'test_suite': { # By default, tests only run at final step
                    'weight_analysis': {'period': 2**20},
                    'sanity_check': {'T': 200, 'period': 2**20},
                    'individual_neuron_activities': {'T': 200, 'period': 2**20},
                    'fit_internal_representation': {'T': 200, 'period': 2**20},
                  },
    'rescale_s_dot': False,
    'rescale_s_norms': False,
}




class relu_D4:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4, 10]:
                params = deepcopy(default_D4_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D4/relu/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


class dale_D4_inhib_frac:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [1000, ]
        algs, lrs, n_epochs = ['sgd'], [1e0], [10000, ]
        T=4

        for a, l, e in zip(algs, lrs, n_epochs):
            for inhib_frac in [.5, .4, .25, .1]:
                params = deepcopy(default_D4_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_type'] = 'DaleNet'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D4/dale/inhib_frac_{}/'.format(inhib_frac)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                params['net_params']['inhib_proportion'] = inhib_frac
                params['net_params']['l2_penalty'] = 0
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

if __name__ == '__main__':
    # This part is for efficient multi thread logging
    n_threads = 4
    n_seeds = 4
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = Pool(n_threads, initializer=install_mp_handler)

    # pool.map(relu_D4(), range(n_seeds))
    pool.map(dale_D4_inhib_frac(), range(n_seeds))
