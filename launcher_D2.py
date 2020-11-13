from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch


default_D2_pars = {
    'net_type': 'many_channels',
    'net_params': { 'train_ed': False,
                    'device_name': 'cuda',
                    'n': 1000,
                    'n_channels': 2,
                    'saturations': [-1e8, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'is_switch': False,
                    'activation_type': 'ReLU',
                    },
    'sampler_params': {
                    'n_channels': 2,
                    'epoch_length': 3,
                    'decays': [.995, 0.993],
                    'scales': [1., 1.],
                    'batch_size': 1024,
                    'is_switch': False
                    },
    'train_params': {'train_ed': False,
                    'train_d_only': False,
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


class linear_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [3,10]:
                params = deepcopy(default_D2_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D2/linear/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['sgd'], [5e-1], [5000]
        # algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            # for T in [3,10]:
            for T in [3]:
                params = deepcopy(default_D2_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D2/relu/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                params['test_suite']['fit_internal_representation']['batch_size'] = 512
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D2_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['sgd'], [1e-1], [ 5000]
        algs, lrs, n_epochs = ['adam'], [1e-3], [ 5000]

        for a, l, e in zip(algs, lrs, n_epochs):
                params = deepcopy(default_D2_pars)
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D2/relu_avg/'
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'stop_loss': 1e-8, 'loss_name': 'avg_d2'})
                params['test_suite']['fit_internal_representation']['batch_size'] = 512
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

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
        T = 4

        for a, l, e in zip(algs, lrs, n_epochs):
            for ed_init in ['random', 'support_same', 'support_disjoint', 'support_random_e','support_same_with_overlap']:
                params = deepcopy(default_D2_pars)
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


class dale_D2:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['sgd'], [1e0], [10000, ]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4]:
                params = deepcopy(default_D2_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_type'] = 'DaleNet'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D2/dale/{}/T_{}/'.format(a, T)
                params['net_params']['inhib_proportion'] = .25
                params['net_params']['l2_penalty'] = 0.
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class dale_D2_inhib_frac:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['sgd'], [1e0], [7000, ]
        T = 4

        for a, l, e in zip(algs, lrs, n_epochs):
            for inhib_frac in [.5, .25, .1,]:
                params = deepcopy(default_D2_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_type'] = 'DaleNet'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D2/dale/inhib_frac_{}/'.format(inhib_frac)
                params['net_params']['inhib_proportion'] = inhib_frac
                params['net_params']['l2_penalty'] = 0.
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D2_train_d:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['sgd'], [5e-1], [5000]
        T = 3
        for a, l, e in zip(algs, lrs, n_epochs):
            params = deepcopy(default_D2_pars)
            params['sampler_params']['epoch_length'] = T
            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['save_folder'] = 'out/D2/relu_D2_train_d/'
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'train_d_only': True})
            params['test_suite']['fit_internal_representation']['batch_size'] = 512
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()




class sigmoid_D2_batch:
    def __init__(self):
        pass

    def __call__(self, seed):

        algs, n_epochs, lr_list = ['adam'], [10000], [5e-5]
        train_d = True

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_bias in [True, False]:
                for sig_slope in [50.]:#1.]:
                    for sig_thresh in [.1,]:#, .1]:
                        params = deepcopy(default_D2_pars)

                        params['sampler_params']['decays'] = [0.8, 0.75]
                        params['sampler_params']['batch_size'] = 512
                        params['sampler_params']['epoch_length'] = 10
                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                        params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'batch'})

                        params['net_params']['activation_type'] = 'Sigmoid'
                        params['net_params']['sigmoid_threshold'] = sig_thresh
                        params['net_params']['sigmoid_slope'] = sig_slope
                        params['net_params']['sigmoid_random_bias'] = False
                        params['net_params']['sigmoid_train_bias'] = train_bias
                        n = 1000
                        params['net_params']['n'] = n
                        params['net_params']['save_folder'] = 'out/D2_sigmoid_batch/n_{}_slope_{}_thresh_{}_train_bias_{}/'.format(n, sig_slope, sig_thresh, train_bias)
                        # if params['net_params']['sigmoid_random_bias']:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                        # if train_bias:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                        # print(d_scale, e_scale)

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        params['test_suite']['fit_internal_representation']['batch_size'] = 128
                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()


class sigmoid_D2_avg:
    def __init__(self):
        pass

    def __call__(self, seed):

        algs, n_epochs, lr_list = ['adam'], [1000], [5e-4]
        train_d = True

        for a, e, l in zip(algs, n_epochs, lr_list):
            # for train_bias in [True, False]:
            for train_bias in [False]:
                for sig_slope in [50.]:#1.]:
                    for sig_thresh in [.1,]:#, .1]:
                        for n in [1024]: #256, 1024,
                            params = deepcopy(default_D2_pars)

                            params['sampler_params']['decays'] = [0.8, 0.75]
                            params['sampler_params']['batch_size'] = 256
                            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                            params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'avg_d2'})

                            params['net_params']['activation_type'] = 'Sigmoid'
                            params['net_params']['sigmoid_threshold'] = sig_thresh
                            params['net_params']['sigmoid_slope'] = sig_slope
                            params['net_params']['sigmoid_random_bias'] = False
                            params['net_params']['sigmoid_train_bias'] = train_bias
                            # n = 1000
                            params['net_params']['n'] = n
                            params['net_params']['save_folder'] = 'out/D2_sigmoid_avg/n_{}/'.format(n)
                            # params['net_params']['save_folder'] = 'out/D2_sigmoid_avg/n_{}_slope_{}_thresh_{}_train_bias_{}/'.format(n, sig_slope, sig_thresh, train_bias)
                            # if params['net_params']['sigmoid_random_bias']:
                            #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                            # if train_bias:
                            #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                            # print(d_scale, e_scale)

                            # for test_name in params['test_suite'].keys():
                            #     params['test_suite'][test_name]['period'] = 1000
                            #
                            # params['test_suite']['fit_internal_representation']['batch_size'] = 128
                            params['test_suite'] = {'sanity_check': {'T': 200, 'period': 1000}, 'error_realtime': {'T': 200, 'period': 1000}, 'individual_neuron_activities': {'T': 200, 'period': 1000},
                                                            'fit_internal_representation': {'T': 200, 'period': 1000},}
                            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                                json.dump(params, f, indent=4)

                            main(params, seed)
                            gc.collect()


class relu_D2_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [8e-5, 2e-1], [4000, 10000], [10, 4], [5, 3]

        # algs, lrs, n_epochs, Ts = ['adam',], [8e-5], [2000,], [5]
        # algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [2.]
        # algs, lrs, n_epochs, Ts, Ss = ['sgd'], [1e-1], [5000], [3], [1.]
        algs, lrs, n_epochs, Ts, Ss = ['adam'], [5e-4], [1000], [3], [2.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            for n in [1024]:
                params = deepcopy(default_D2_pars)
                params['sampler_params']['epoch_length'] = T
                params['sampler_params']['scales'] = [s, s]
                params['sampler_params']['decays'] = [0.8, 0.75]
                params['net_params']['init_vectors_scales']: [1., 1.] # try to make the prefered scale equal to 2
                params['sampler_params']['batch_size'] = 256
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d2'})#, 'loss_name': 'avg_d1'})
                params['net_params']['save_folder'] = 'out/D2_relu_avg/n_{}/'.format(n)
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['n'] = n

                params['test_suite'] = {'sanity_check': {'T': 200, 'period': 1000}, 'error_realtime': {'T': 200, 'period': 1000}, 'individual_neuron_activities': {'T': 200, 'period': 1000},
                                                'fit_internal_representation': {'T': 200, 'period': 1000},}
                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

if __name__ == '__main__':
    # This part is for efficient multi thread logging
    n_threads = 4
    start_seed = 0
    n_seeds = 16
    # start_seed = 9
    # n_seeds = 4

    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = Pool(n_threads, initializer=install_mp_handler)

    ### For T=3, Adam gets much better loss, but poorly generalizes.
    # pool.map(linear_D2(), range(n_seeds))
    # pool.map(relu_D2(), range(start_seed, start_seed+n_seeds))
    # pool.map(relu4_D2(), range(n_seeds))
    # pool.map(dale_D2_inhib_frac(), range(n_seeds))
    # pool.map(relu_D2_supports(), range(n_seeds))
    # pool.map(relu_D2_train_d(), range(n_seeds))
    pool.map(relu_D2_avg(), range(n_seeds))
    # pool.map(sigmoid_D2_avg(), range(start_seed, start_seed+n_seeds))
    # pool.map(sigmoid_D2_batch(), range(n_seeds))
    # pool.map(relu_D2_avg(), range(n_seeds))
