'''
We provide a selected subset of experiments that allow reproducing the most relevant results of the original paper,
without having to rerun the extensive exploration work associated. 

'''
from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch

default_params = {
    'net_type': 'many_channels',
    'net_params': { 'train_ed': False,
                    'device_name': 'cuda',
                    'n': 1024,
                    'n_channels': 2,
                    'saturations': [0, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'is_switch': False,
                    'activation_type': 'ReLU',
                    'sigmoid_random_bias': False,
                    'sigmoid_threshold': 0.,
                    'sigmoid_train_bias': False,
                    'sigmoid_slope': 1,
                    # 'sigmoid_slope': 1.,
                    },
    'sampler_params': {
                    'n_channels': 2,
                    'epoch_length': 3,
                    'decays': [.995, 0.99],
                    'scales': [1., 1.],
                    'batch_size': 256,
                    'is_switch': False
                    },
    'train_params': {'train_ed': False,
                    'train_d_only': False,
                    'loss_name': 'batch',
                    'optimizer_name': 'sgd',
                    'lr': 1e-1,
                    'normalize_grad': False,
                    'n_epochs': 5000,
                    'stop_loss': 1e-9,
                    },
    'test_suite': { # By default, tests only run at final step
                    'weight_analysis': {'period': 2**20},
                    'sanity_check': {'T': 200, 'period': 2**20},
                    'error_realtime': {'T': 200, 'period': 2**20},
                    'fit_internal_representation': {'T': 200, 'period': 2**20},
                  },
    'rescale_s_dot': False,
    'rescale_s_norms': False,
}

class relu_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd',], [3e-4, 1], [1000, 10000]

        decay_modes = {'fast_decay': [.9, .85], 'slow_decay': [.992, .99]}
        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [10, 3]:
                for name, decays in decay_modes.items():
                    params = deepcopy(default_params)
                    params['sampler_params']['epoch_length'] = T
                    params['net_params']['save_folder'] = 'out/D2/relu/{}_T_{}_{}/'.format(name, T, a)
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                    params['sampler_params']['decays'] = decays

                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)
                    main(params, seed)
                    gc.collect()

class relu_D2_avg_fast_decay:
    # sgd is a bit finicky with larger ranges
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd',], [5e-4, 1e-1,], [3000, 5000]

        d = {'fast_decay': ([.9, .85], 8)}

        for a, l, e in zip(algs, lrs, n_epochs):
            for name, (decays, spread) in d.items():
                params = deepcopy(default_params)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d2'})
                params['net_params']['save_folder'] = f'out/D2_relu_avg/{name}/{a}/'
                params['net_params']['saturations'] = [0, 1e8]
                params['sampler_params']['decays'] = decays
                # sampler_params are accessed by the loss when executed, so will see that !
                params['sampler_params']['avg_loss_range'] = spread

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D2_avg_slow_decay:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd',], [5e-4, 5e-3,], [3000, 3000]
        d = {'slow_decay': ([.992, .99], 40)}

        for a, l, e in zip(algs, lrs, n_epochs):
            for name, (decays, spread) in d.items():
                params = deepcopy(default_params)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d2'})
                params['net_params']['save_folder'] = f'out/D2_relu_avg/{name}/{a}/'
                params['net_params']['saturations'] = [0, 1e8]
                params['sampler_params']['decays'] = decays

                # sampler_params are accessed by the loss when executed, so will see that !
                params['net_params']['e'] = spread

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class dale_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [5e-4, 1e0,], [3000, 10000, ]
        decay_modes = {'fast_decay': [.9, .85], 'slow_decay': [.992, .99]}

        for a, l, e in zip(algs, lrs, n_epochs):
            for decay_name, decays in decay_modes.items():
                for T in [10, 3]:
                    
                    params = deepcopy(default_params)
                    params['sampler_params']['epoch_length'] = T
                    params['net_type'] = 'DaleNet'
                    params['net_params']['save_folder'] = 'out/D2_dale/{}_T_{}_{}/'.format(decay_name, T, a)
                    params['net_params']['inhib_proportion'] = .25
                    params['net_params']['l2_penalty'] = 0.
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                    params['sampler_params']['decays'] = decays

                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()

class relu_D2_supports:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]
        # For this experiment, sgd should be used otherwise adam will introduce couplings; 
        # effect is still visible, but not as exact 
        T = 5

        for a, l, e in zip(algs, lrs, n_epochs):
            for ed_init in ['random', 'support_same', 'support_disjoint', 'support_random_e','support_same_with_overlap']:
                params = deepcopy(default_params)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['init_vectors_type'] = ed_init
                params['net_params']['save_folder'] = 'out/D2/relu_support/{}/{}/'.format(ed_init, a)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class sigmoid_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd',], [1e-4, 1], [1000, 10000]
        algs, lrs, n_epochs = ['adam'], [8e-5,], [4000]


        decay_modes = {'fast_decay': [.9, .85], 'slow_decay': [.992, .99]}
        # decay_modes = {'fast_decay': [.9, .85], 'slow_decay': [.992, .99]}
        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [10, 3]:
                for name, decays in decay_modes.items():
                    params = deepcopy(default_params)
                    params['sampler_params']['epoch_length'] = T
                    params['net_params']['save_folder'] = 'out/D2/sigmoid/{}_T_{}_{}/'.format(name, T, a)
                    params['net_params']['activation_type'] = 'Sigmoid'
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                    params['sampler_params']['decays'] = decays

                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)
                    main(params, seed)
                    gc.collect()

class sigmoid_D2_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam'], [8e-5,], [4000]

        decay_modes = {
                'fast_decay': [.9, .85], 
                'slow_decay': [.992, .99]}
        for a, l, e in zip(algs, lrs, n_epochs):
            for name, decays in decay_modes.items():
                # if name == 'slow_decay' and a =='sgd':
                #     l = 5e-3
                params = deepcopy(default_params)
                params['net_params']['save_folder'] = 'out/D2/sigmoid_avg/{}_{}/'.format(name, a)
                params['net_params']['activation_type'] = 'Sigmoid'
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d2'})
                params['sampler_params']['decays'] = decays

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)
                main(params, seed)
                gc.collect()

class relu_D5_avg:
    # sgd is a bit finicky with larger ranges
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam',], [5e-4, ], [5000]

        for a, l, e in zip(algs, lrs, n_epochs):

                params = deepcopy(default_params)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_generic'})
                params['net_params']['save_folder'] = f'out/D5_relu_avg/{a}/'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['n_channels'] = 5
                params['sampler_params']['n_channels'] = 5
                params['sampler_params']['scales'] = [1., 1., 1., 1., 1.]
                params['sampler_params']['decays'] = [.92, .9, .88, .87, .85]
                # sampler_params are accessed by the loss when executed, so will see that !
                params['sampler_params']['avg_loss_range'] = 10

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


if __name__ == '__main__':
    n_threads = 2
    start_seed = 0
    n_seeds = 16


    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = Pool(n_threads, initializer=install_mp_handler)
    # pool.map(relu_D2(), range(start_seed, start_seed+n_seeds))
    # pool.map(relu_D2_avg_fast_decay(), range(n_seeds))
    # pool.map(relu_D2_avg_slow_decay(), range(n_seeds))

    # pool.map(dale_D2(), range(n_seeds))
    # pool.map(relu_D2_supports(), range(n_seeds))
    # pool.map(sigmoid_D2(), range(n_seeds))
    # pool.map(sigmoid_D2_avg(), range(n_seeds))

    # pool.map(relu_D5_avg(), range(n_seeds))

    # # # Once everything ran, can aggregate results:
    from aggregators import aggregate_dale, aggregate_angles_D2, aggregate_distance_to_manifold
    # aggregate_dale(folder='out/D2_dale/slow_decay_T_10_sgd/')
    # aggregate_angles_D2(folder='out/D2/relu/slow_decay_T_10_adam/')

    from pathlib import Path

    named_folders = {
        'adam_relu_D2_avg_slow': 'out/D2_relu_avg/slow_decay/adam/',
        'adam_relu_D2_batch_slow': 'out/D2/relu/slow_decay_T_10_adam/',
        'adam_relu_D2_avg_fast': 'out/D2_relu_avg/fast_decay/adam/',
        'adam_relu_D2_batch_fast': 'out/D2/relu/fast_decay_T_10_adam/',
        'sgd_relu_D2_avg_slow': 'out/D2_relu_avg/slow_decay/sgd/',
        'sgd_relu_D2_batch_slow': 'out/D2/relu/slow_decay_T_10_sgd/',
        'sgd_relu_D2_avg_fast': 'out/D2_relu_avg/fast_decay/sgd/',
        'sgd_relu_D2_batch_fast': 'out/D2/relu/fast_decay_T_10_sgd/',
        'adam_relu_D5_avg': 'out/D5_relu_avg/adam/',
    }
    aggregate_distance_to_manifold(named_folders)