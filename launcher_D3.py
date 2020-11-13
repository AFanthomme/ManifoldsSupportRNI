from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch


default_D3_pars = {
    'net_type': 'many_channels',
    'net_params': { 'train_ed': False,
                    'device_name': 'cuda',
                    'n': 1000,
                    'n_channels': 3,
                    'saturations': [-1e8, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'is_switch': False,
                    'activation_type': 'ReLU',
                    },
    'sampler_params': {
                    'n_channels': 3,
                    'epoch_length': 3,
                    'decays': [.995, 0.992, 0.992],
                    'scales': [1., 1., 1.],
                    'batch_size': 1024,
                    'is_switch': False,
                    },
    'train_params': {'train_ed': False,
                    'loss_name': 'batch',
                    'optimizer_name': 'sgd',
                    'lr': 1e-1,
                    'n_epochs': 5000,
                    'stop_loss': 1e-9,
                    'train_d_only': False,
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




class relu_D3:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4, 10]:
                params = deepcopy(default_D3_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D3/relu/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()



class dale_D3:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [1000, ]
        algs, lrs, n_epochs = ['sgd'], [1e0], [10000, ]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4]:
                params = deepcopy(default_D3_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_type'] = 'DaleNet'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D3/dale/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                params['net_params']['l2_penalty'] = 0
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class dale_D3_inhib_frac:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [3e-4, 5e-1], [1000, 5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [1000, ]
        algs, lrs, n_epochs = ['sgd'], [1e0], [10000, ]
        T=4

        for a, l, e in zip(algs, lrs, n_epochs):
            for inhib_frac in [.5, .4, .25, .1]:
                params = deepcopy(default_D3_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_type'] = 'DaleNet'
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['save_folder'] = 'out/D3/dale/inhib_frac_{}/'.format(inhib_frac)
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




class sigmoid_D3_batch:
    def __init__(self):
        pass

    def __call__(self, seed):

        algs, n_epochs, lr_list = ['adam'], [10000], [5e-5]
        train_d = True

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_bias in [True, False]:
                for sig_slope in [50.]:#1.]:
                    for sig_thresh in [.1,]:#, .1]:
                        params = deepcopy(default_D3_pars)

                        params['sampler_params']['decays'] = [0.8, 0.75, .75]
                        params['sampler_params']['batch_size'] = 1024
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
                        params['net_params']['save_folder'] = 'out/D3_sigmoid_batch/n_{}_slope_{}_thresh_{}_train_bias_{}/'.format(n, sig_slope, sig_thresh, train_bias)
                        # if params['net_params']['sigmoid_random_bias']:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                        # if train_bias:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                        # print(d_scale, e_scale)
                        params['test_suite'] = { # By default, tests only run at final step
                                            'weight_analysis': {'period': 2**20},
                                            'sanity_check': {'T': 200, 'period': 2**20},
                                            'fit_internal_representation': {'T': 200, 'period': 2**20},
                                          }

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()

class relu_D3_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam'], [5e-4], [5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for n in [1024]:
                params = deepcopy(default_D3_pars)
                params['sampler_params']['decays'] = [0.8, 0.77, 0.75]
                params['net_params']['saturations'] = [0, 1e8]
                params['sampler_params']['batch_size'] = 4096
                params['net_params']['n'] = n
                params['net_params']['save_folder'] = 'out/D3_relu_avg/n_{}/'.format(n)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_generic'})
                params['test_suite'] = { # By default, tests only run at final step
                                    'weight_analysis': {'period': 1000},
                                    'sanity_check': {'T': 200, 'period': 1000},
                                  }

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D5_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam'], [5e-4], [5000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for n in [1024]:
                params = deepcopy(default_D3_pars)
                params['sampler_params']['scales'] = [1., 1., 1., 1., 1.]
                params['sampler_params']['decays'] = [0.8, 0.77, 0.75, 0.76, 0.79]
                params['net_params']['saturations'] = [0, 1e8]
                params['sampler_params']['batch_size'] = 4096
                params['net_params']['n'] = n
                params['net_params']['n_channels'] = 5
                params['sampler_params']['n_channels'] = 5
                params['net_params']['init_vectors_scales']= [1., 1.]
                params['net_params']['save_folder'] = 'out/D5_relu_avg/n_{}/'.format(n)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_generic'})
                params['test_suite'] = { # By default, tests only run at final step
                                    'weight_analysis': {'period': 1000},
                                    'sanity_check': {'T': 200, 'period': 1000},
                                    'error_realtime': {'T': 200, 'period': 1000},
                                    'fit_internal_representation': {'T': 200, 'period': 1000, 'batch_size': 128}
                                  }

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


class sigmoid_D3_avg:
    def __init__(self):
        pass

    def __call__(self, seed):

        algs, n_epochs, lr_list = ['adam'], [10000], [5e-4]
        train_d = True
        train_bias = False


        for a, e, l in zip(algs, n_epochs, lr_list):
            # for n in [8192]:
            for n in [1024]:
                for sig_slope in [5.]:#1.]:
                    for sig_thresh in [0.,]:#, .1]:
                        params = deepcopy(default_D3_pars)

                        params['sampler_params']['decays'] = [0.8, 0.77, .75]
                        params['sampler_params']['batch_size'] = 4096

                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                        params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'avg_generic'})

                        params['net_params']['activation_type'] = 'Sigmoid'
                        params['net_params']['sigmoid_threshold'] = sig_thresh
                        params['net_params']['sigmoid_slope'] = sig_slope
                        params['net_params']['sigmoid_random_bias'] = False
                        params['net_params']['sigmoid_train_bias'] = train_bias
                        params['net_params']['init_vectors_scales']= [1., 1.]

                        params['net_params']['n'] = n
                        params['net_params']['save_folder'] = 'out/D3_sigmoid_avg/n_{}/'.format(n)
                        # if params['net_params']['sigmoid_random_bias']:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                        # if train_bias:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                        # print(d_scale, e_scale)
                        params['test_suite'] = { # By default, tests only run at final step
                                            # 'weight_analysis': {'period': 1000},
                                            'sanity_check': {'T': 200, 'period': 1000},
                                            'error_realtime': {'T': 200, 'period': 1000},
                                            'fit_internal_representation': {'T': 200, 'period': 1000, 'batch_size': 128}
                                          }

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()

class sigmoid_D5_avg:
    def __init__(self):
        pass

    def __call__(self, seed):

        algs, n_epochs, lr_list = ['adam'], [10000], [5e-4]
        train_d = True
        train_bias = False


        for a, e, l in zip(algs, n_epochs, lr_list):
            # for n in [8192]:
            for n in [1024]:
                for sig_slope in [5.]:#1.]:
                    for sig_thresh in [0.,]:#, .1]:
                        params = deepcopy(default_D3_pars)

                        params['sampler_params']['scales'] = [1., 1., 1., 1., 1.]
                        params['sampler_params']['decays'] = [0.8, 0.77, 0.75, 0.76, 0.79]
                        params['net_params']['n_channels'] = 5
                        params['sampler_params']['n_channels'] = 5
                        params['sampler_params']['batch_size'] = 4096

                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                        params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'avg_generic'})

                        params['net_params']['activation_type'] = 'Sigmoid'
                        params['net_params']['sigmoid_threshold'] = sig_thresh
                        params['net_params']['sigmoid_slope'] = sig_slope
                        params['net_params']['sigmoid_random_bias'] = False
                        params['net_params']['sigmoid_train_bias'] = train_bias
                        params['net_params']['init_vectors_scales']= [1., 1.]

                        params['net_params']['n'] = n
                        params['net_params']['save_folder'] = 'out/D5_sigmoid_avg/n_{}/'.format(n)
                        # if params['net_params']['sigmoid_random_bias']:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                        # if train_bias:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                        # print(d_scale, e_scale)
                        params['test_suite'] = { # By default, tests only run at final step
                                            # 'weight_analysis': {'period': 1000},
                                            'sanity_check': {'T': 200, 'period': 1000},
                                            'error_realtime': {'T': 200, 'period': 1000},
                                            'fit_internal_representation': {'T': 200, 'period': 1000, 'batch_size': 128}
                                          }

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()

# class sigmoid_D3_avg_non_linear_decoder:
#     def __init__(self):
#         pass
#
#     def __call__(self, seed):
#
#         algs, n_epochs, lr_list = ['adam'], [10000], [1e-3]
#         # algs, n_epochs, lr_list = ['sgd'], [10000], [1e-2]
#         train_d = True
#         train_bias = False
#
#
#         for a, e, l in zip(algs, n_epochs, lr_list):
#             # for n in [8192]:
#             for n in [1024]:
#                 for sig_slope in [1.]:#1.]:
#                     for sig_thresh in [0,]:#, .1]:
#                         params = deepcopy(default_D3_pars)
#
#                         params['net_type'] = 'NonLinearDecoder'
#                         params['sampler_params']['decays'] = [0.8, 0.77, .75]
#                         params['sampler_params']['batch_size'] = 1024
#
#                         params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
#                         params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'avg_generic_non_linear_decoder'})
#
#                         params['net_params']['activation_type'] = 'Sigmoid'
#                         params['net_params']['sigmoid_threshold'] = sig_thresh
#                         params['net_params']['sigmoid_slope'] = sig_slope
#                         params['net_params']['sigmoid_random_bias'] = False
#                         params['net_params']['sigmoid_train_bias'] = train_bias
#                         params['net_params']['init_vectors_scales']= [1., 1.]
#                         params['net_params']['init_radius']= 0.
#
#                         params['net_params']['n'] = n
#                         params['net_params']['save_folder'] = 'out/D3_sigmoid_avg_non_linear_decoder/n_{}/'.format(n)
#                         # if params['net_params']['sigmoid_random_bias']:
#                         #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
#                         # if train_bias:
#                         #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
#                         # print(d_scale, e_scale)
#                         params['test_suite'] = { # By default, tests only run at final step
#                                             # 'weight_analysis': {'period': 1000},
#                                             'sanity_check': {'T': 200, 'period': 100},
#                                             'error_realtime': {'T': 200, 'period': 100}
#                                           }
#
#                         for test_name in params['test_suite'].keys():
#                             params['test_suite'][test_name]['period'] = 1000
#
#                         os.makedirs(params['net_params']['save_folder'], exist_ok=True)
#                         with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
#                             json.dump(params, f, indent=4)
#
#                         main(params, seed)
#                         gc.collect()

# class relu_D5_avg_non_linear_decoder:
#     def __init__(self):
#         pass
#
#     def __call__(self, seed):
#         algs, lrs, n_epochs = ['adam'], [5e-4], [5000]
#
#         for a, l, e in zip(algs, lrs, n_epochs):
#             for n in [1024]:
#                 params = deepcopy(default_D3_pars)
#                 params['net_type'] = 'NonLinearDecoder'
#                 params['sampler_params']['scales'] = [1., 1., 1., 1., 1.]
#                 params['sampler_params']['decays'] = [0.8, 0.77, 0.75, 0.76, 0.79]
#                 params['net_params']['saturations'] = [0, 1e8]
#                 params['sampler_params']['batch_size'] = 4096
#                 params['net_params']['n'] = n
#                 params['net_params']['n_channels'] = 5
#                 params['sampler_params']['n_channels'] = 5
#                 params['net_params']['save_folder'] = 'out/D5_relu_avg_non_linear_decoder/n_{}/'.format(n)
#                 params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_generic_non_linear_decoder'})
#                 params['test_suite'] = { # By default, tests only run at final step
#                                     'weight_analysis': {'period': 1000},
#                                     'sanity_check': {'T': 200, 'period': 1000},
#                                     'error_realtime': {'T': 200, 'period': 1000},
#                                     'fit_internal_representation': {'T': 200, 'period': 1000, 'batch_size': 128}
#                                   }
#
#                 os.makedirs(params['net_params']['save_folder'], exist_ok=True)
#                 with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
#                     json.dump(params, f, indent=4)
#
#                 main(params, seed)
#                 gc.collect()



if __name__ == '__main__':
    # This part is for efficient multi thread logging
    # import warnings
    # warnings.filterwarnings("ignore")
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    n_threads = 4
    n_seeds = 4
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = Pool(n_threads, initializer=install_mp_handler)

    # pool.map(relu_D3(), range(n_seeds))
    # pool.map(dale_D3(), range(n_seeds))
    # pool.map(dale_D3_inhib_frac(), range(n_seeds))
    # pool.map(sigmoid_D3_batch(), range(n_seeds))
    # pool.map(sigmoid_D3_avg(), range(n_seeds))
    # pool.map(relu_D3_avg(), range(n_seeds))
    # pool.map(relu_D5_avg(), range(n_seeds))
    # pool.map(sigmoid_D3_avg_non_linear_decoder(), range(n_seeds))
    # pool.map(relu_D5_avg_non_linear_decoder(), range(n_seeds))
    pool.map(sigmoid_D5_avg(), range(n_seeds))
