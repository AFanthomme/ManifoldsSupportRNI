from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch


default_D1_pars = {
    'net_type': 'many_channels',
    'net_params': { 'train_ed': False,
                    'device_name': 'cuda',
                    'n': 1000,
                    'n_channels': 1,
                    'saturations': [-1e8, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'init_vectors_overlap': 0.,
                    'init_vectors_scales': [1,1],
                    'is_switch': False,
                    'activation_type': 'ReLU',
                    },
    'sampler_params': {
                    'n_channels': 1,
                    'epoch_length': 3,
                    'decays': [.995, ],
                    'scales': [1., ],
                    'batch_size': 1024,
                    'is_switch': False,
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
                    # 'individual_neuron_activities': {'T': 200, 'period': 2**20},
                    'fit_internal_representation': {'T': 200, 'period': 2**20},
                  },
    'rescale_s_dot': False,
    'rescale_s_norms': False,
}


class linear_D1_batch_T:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4,2,3,]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D1/linear_batch_T/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D1_batch_T:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4,2,3,]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D1/relu_batch_T/{}/T_{}/'.format(a, T)
                params['net_params']['saturations'] = [0, 1e8]

                params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


class linear_D1_avg_T:
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]

        for a, l, e in zip(algs, lrs, n_epochs):
            for T in [4,2,3,]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D1/linear_avg_T/{}/T_{}/'.format(a, T)
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d1'})
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


class relu_D1_avg:
    # No need to vary T, since the loss does not use it
    def __init__(self):
        pass

    def __call__(self, seed):
        algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]

        for a, l, e in zip(algs, lrs, n_epochs):
            params = deepcopy(default_D1_pars)

            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['save_folder'] = 'out/D1/relu_avg/{}/'.format(a)
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d1'})

            params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()


class linear_D1_cvg_experimental:
    def __init__(self):
        pass

    def __call__(self, seed):
        n_epochs = 10000



        s_list = [.1, 1., 10]
        skimmed_lr_list = [
            [3e-2, 1e-2],
            [7e-1, 6e-1],
            [1e-1, 8e-2],
        ]

        for s, lr_list in zip(s_list, lr_lists):
            for lr in lr_list:
                params = deepcopy(default_D1_pars)
                params['net_params']['save_folder'] = 'out/D1/linear_cvg_experimental/s_{}/lr_{}/'.format(s, lr)
                params['train_params'].update({'lr': lr, 'n_epochs': n_epochs, 'loss_name': 'avg_d1'})
                params['sampler_params']['scales'] = [s]
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = n_epochs // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()


class relu_D1_cvg_experimental:
    def __init__(self):
        pass

    def __call__(self, seed):
        n_epochs = 100000
        s_list = [.05, .1, .3, .6, 1, 2, 4, 10]
        lr_lists = [
            [6e-4, 4e-4], # s=.05, (drop is around 6e-4)
            [3e-3, 1e-3], # s=0.1, (drop is around 2e-3)
            [3e-2, 1e-2], # s=0.3, (drop is around 2e-2)
            [8e-2, 3e-2], # s=0.6, (drop is around 9e-2)
            [3e-1, 1e-1], # s=1, (drop is around 2e-1)
            [3e-1, 1e-1], # s=2, (drop is around 2e-1)
            [3e-1, 1e-1], # s=4, (drop is around 2e-1)
            [3e-1, 1e-1], # s=10, (drop is around 2e-1)
            [3e-1, 1e-1], # s=20, (drop is around 2e-1)
        ]

        # for s, lr_list in zip(s_list, lr_lists):
        for s, lr_list in zip([.6], lr_lists):
            for lr in lr_list:
                params = deepcopy(default_D1_pars)
                params['net_params']['save_folder'] = 'out/D1/relu_cvg_experimental/s_{}/lr_{}/'.format(s, lr)
                params['net_params']['saturations'] = [0, 1e8]
                params['train_params'].update({'lr': lr, 'n_epochs': n_epochs, 'loss_name': 'avg_d1'})
                params['sampler_params']['scales'] = [s]
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = n_epochs // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_D1_for_dots_figure:
    def __init__(self):
        pass

    def __call__(self, seed):
        n_epochs = 100000
        s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
        lr_list = [4e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]

        for s, lr in zip(s_list, lr_list):
            params = deepcopy(default_D1_pars)
            params['net_params']['save_folder'] = 'out/D1/relu_dots/s_{}/'.format(s)
            params['net_params']['saturations'] = [0, 1e8]
            params['train_params'].update({'lr': lr, 'n_epochs': n_epochs, 'loss_name': 'avg_d1', 'stop_loss': 1e-7})
            params['sampler_params']['scales'] = [s]
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = n_epochs // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()

class relu_D1_for_hessian:
    def __init__(self):
        pass

    def __call__(self, seed):
        n_epochs = 100000
        s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
        lr_list = [4e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]

        for s, lr in zip(s_list, lr_list):
            params = deepcopy(default_D1_pars)
            params['net_params']['save_folder'] = 'out/D1/relu_hessian/s_{}/'.format(s)
            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['n'] = 50
            params['train_params'].update({'lr': lr, 'n_epochs': n_epochs, 'loss_name': 'avg_d1', 'stop_loss': 1e-7})
            params['sampler_params']['scales'] = [s]
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = n_epochs // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()



class dale_D1_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]
        algs, lrs, n_epochs = ['sgd'], [1e-1], [5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [5000]
        T = 4

        for a, l, e in zip(algs, lrs, n_epochs):
            params = deepcopy(default_D1_pars)
            params['sampler_params']['epoch_length'] = T
            params['net_params']['save_folder'] = 'out/D1/dale_avg/{}/'.format(a)
            params['net_params']['saturations'] = [0, 1e8]
            params['net_type'] = 'DaleNet'
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d1'})

            params['test_suite'] = { # By default, tests only run at final step
                'weight_analysis': {'period': 2**20},
                'sanity_check': {'T': 200, 'period': 2**20},
                'individual_neuron_activities': {'T': 200, 'period': 2**20},
                'fit_internal_representation': {'T': 200, 'period': 2**20},
              }
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()

class dale_D1_avg_inhib_frac:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]
        algs, lrs, n_epochs = ['sgd'], [1e-1], [5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [5000]
        T = 4


        for a, l, e in zip(algs, lrs, n_epochs):
            for inhib_frac in [.5, .4, .25, .1, .15, .05, 0.]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D1/dale_avg/inhib_frac_{}/{}/'.format(inhib_frac, a)
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['inhib_proportion'] = inhib_frac
                params['net_type'] = 'DaleNet'
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'loss_name': 'avg_d1'})

                params['test_suite'] = { # By default, tests only run at final step
                    'weight_analysis': {'period': 2**20},
                    'sanity_check': {'T': 200, 'period': 2**20},
                    'individual_neuron_activities': {'T': 200, 'period': 2**20},
                    'fit_internal_representation': {'T': 200, 'period': 2**20},
                    'relu_D1_currents': {'T': 200, 'period': 2**20},
                  }


                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class dale_D1_batch_inhib_frac:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]
        algs, lrs, n_epochs = ['sgd'], [1e0], [5000]
        # algs, lrs, n_epochs = ['adam'], [1e-3], [5000]
        T = 4


        for a, l, e in zip(algs, lrs, n_epochs):
            for inhib_frac in [.5, .4, .25, .1, .15, .05]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['net_params']['save_folder'] = 'out/D1/dale_batch/inhib_frac_{}/{}/'.format(inhib_frac, a)
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['inhib_proportion'] = inhib_frac
                params['net_params']['l2_penalty'] = 0.
                params['net_type'] = 'DaleNet'
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})

                params['test_suite'] = { # By default, tests only run at final step
                    'weight_analysis': {'period': 2**20},
                    'sanity_check': {'T': 200, 'period': 2**20},
                    'individual_neuron_activities': {'T': 200, 'period': 2**20},
                    'fit_internal_representation': {'T': 200, 'period': 2**20},
                  }
                for test_name in params['test_suite'].keys():
                    params['test_suite'][test_name]['period'] = e // 4

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

class relu_biased_D1:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs = ['adam', 'sgd'], [1e-4, 1e-1], [1000, 2000]
        # algs, lrs, n_epochs = ['sgd'], [1e-2], [5000]
        algs, lrs, n_epochs = ['adam'], [1e-4], [5000]
        T = 10

        for a, l, e in zip(algs, lrs, n_epochs):
            params = deepcopy(default_D1_pars)
            params['sampler_params']['epoch_length'] = T
            params['sampler_params']['batch_size'] = 2048
            params['net_params']['save_folder'] = 'out/D1/relu_biased/{}/'.format(a)
            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['init_vectors_type'] = 'random_biased'
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})

            params['test_suite'] = { # By default, tests only run at final step
                'weight_analysis': {'period': 2**20},
                'sanity_check': {'T': 200, 'period': 2**20},
                'individual_neuron_activities': {'T': 200, 'period': 2**20},
                'fit_internal_representation': {'T': 200, 'period': 2**20},
              }
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 4

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()



class relu_D1_train_d_vary_norms:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [1e-4, 2e-1], [2000, 5000], [10, 4], [1., 1.]

        # algs, lrs, n_epochs, Ts, Ss = ['adam',], [1e-4], [1000,], [5], [1.]
        algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [8000], [3], [1.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            for train_d in [True, False]:
                for d_scale in [.1, 0., 1., 10.]:
                    for e_scale in [1., .1, 10.]:
                        if d_scale * e_scale > 1:
                            l *= d_scale * e_scale
                        params = deepcopy(default_D1_pars)
                        params['sampler_params']['epoch_length'] = T
                        params['sampler_params']['scales'] = [s]
                        params['sampler_params']['decays'] = [0.99]
                        params['sampler_params']['batch_size'] = 4096
                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
                        # if small_init:
                        #     params['train_params'].update({'train_d_only': True, 'd_init': 'small'})#, 'loss_name': 'avg_d1'})
                        # else:
                        params['train_params'].update({'train_d_only': train_d, 'd_init': 'normal'})#, 'loss_name': 'avg_d1'})
                        if train_d:
                            params['net_params']['save_folder'] = 'out/D1_relu_training_ed/training_d_only/{}_T_{}/d_{}_e_{}/'.format(train_d, a, T, d_scale, e_scale)
                        else:
                            params['net_params']['save_folder'] = 'out/D1_relu_training_ed/training_d_only/{}_T_{}/d_{}_e_{}/'.format(train_d, a, T, d_scale, e_scale)
                        params['net_params']['saturations'] = [0, 1e8]
                        params['net_params']['n'] = 1000
                        # print(d_scale, e_scale)
                        params['net_params'].update({'init_vectors_scales': [d_scale, e_scale, ]})

                        params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = e // 5

                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()

class relu_D1_train_ed_vary_norms:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [1e-4, 2e-1], [2000, 5000], [10, 4], [1., 1.]

        # algs, lrs, n_epochs, Ts, Ss = ['adam',], [1e-4], [1000,], [5], [1.]
        algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [10000], [3], [1.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            for train_ed in [True]:
                for d_scale in [.1, 0., 1., 10.]:
                    for e_scale in [1., .1, 10.]:
                        params = deepcopy(default_D1_pars)
                        params['sampler_params']['epoch_length'] = T
                        params['sampler_params']['scales'] = [s]
                        params['sampler_params']['decays'] = [0.99]
                        params['sampler_params']['batch_size'] = 4096
                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
                        # if small_init:
                        #     params['train_params'].update({'train_d_only': True, 'd_init': 'small'})#, 'loss_name': 'avg_d1'})
                        # else:
                        params['train_params'].update({'train_ed': train_ed})#, 'loss_name': 'avg_d1'})
                        params['net_params']['save_folder'] = 'out/D1_relu_training_ed/training_e_and_d/{}_T_{}/d_{}_e_{}/'.format(train_ed, a, T, d_scale, e_scale)
                        params['net_params']['saturations'] = [0, 1e8]
                        params['net_params']['n'] = 1000
                        # print(d_scale, e_scale)
                        params['net_params'].update({'init_vectors_scales': [d_scale, e_scale, ]})

                        params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = e // 5

                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()

class relu_D1_train_d_vary_s:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        algs, n_epochs, Ts = ['sgd'], [30000], [3]
        Ss = [.5, .1]
        lrs = [1e-1, 8e-2]

        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            for train_d in [True, False]:
                for l, s in zip(lrs, Ss):
                    params = deepcopy(default_D1_pars)
                    params['sampler_params']['epoch_length'] = T
                    params['sampler_params']['scales'] = [s]
                    params['sampler_params']['decays'] = [0.995]
                    params['sampler_params']['batch_size'] = 4096
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
                    # if small_init:
                    #     params['train_params'].update({'train_d_only': True, 'd_init': 'small'})#, 'loss_name': 'avg_d1'})
                    # else:
                    params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7})#, 'loss_name': 'avg_d1'})

                    params['net_params']['save_folder'] = 'out/D1_relu_training_d/training_d_{}/{}_T_{}/s_{}/'.format(train_d, a, T, s)
                    params['net_params']['saturations'] = [0, 1e8]
                    params['net_params']['n'] = 1000
                    # print(d_scale, e_scale)

                    params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
                    for test_name in params['test_suite'].keys():
                        params['test_suite'][test_name]['period'] = 1000

                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()


class relu_D1_proxy_train_d_vary_s:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        algs, n_epochs = ['sgd'], [100000]
        s_list = [.1, .3, .6, 1, 2, 4, 10]
        lr_list = [1e-3, 1e-2, 3e-2, 1e-1, 1e-1, 1e-1, 1e-1]

        for a, e in zip(algs, n_epochs):
            for train_d in [True, False]:
                for l, s in zip(lr_list, s_list):
                    params = deepcopy(default_D1_pars)

                    params['sampler_params']['scales'] = [s]
                    params['sampler_params']['decays'] = [0.995]
                    params['sampler_params']['batch_size'] = 4096
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
                    # if small_init:
                    #     params['train_params'].update({'train_d_only': True, 'd_init': 'small'})#, 'loss_name': 'avg_d1'})
                    # else:
                    params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7, 'loss_name': 'avg_d1'})

                    params['net_params']['save_folder'] = 'out/D1_relu_proxy_training_d/training_d_{}/s_{}/'.format(train_d, s)
                    params['net_params']['saturations'] = [0, 1e8]
                    params['net_params']['n'] = 1000
                    # print(d_scale, e_scale)

                    params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
                    for test_name in params['test_suite'].keys():
                        params['test_suite'][test_name]['period'] = 5000

                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()


class sigmoid_D1_avg_threshold:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        # algs, n_epochs, lr_list = ['sgd'], [10000], [1e-3]
        algs, n_epochs, lr_list = ['adam'], [10000], [5e-4]

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_d in [True, False]:
                for sig_thresh in [2., -2., 5., -5.]:#0.]:
                    params = deepcopy(default_D1_pars)

                    params['sampler_params']['decays'] = [0.8]
                    params['sampler_params']['batch_size'] = 4096
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                    params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7, 'loss_name': 'avg_d1'})

                    params['net_params']['activation_type'] = 'Sigmoid'
                    params['net_params']['sigmoid_threshold'] = sig_thresh
                    params['net_params']['sigmoid_random_bias'] = False
                    params['net_params']['sigmoid_slope'] = 1.
                    params['net_params']['sigmoid_train_bias'] = False
                    params['net_params']['n'] = 1000
                    params['net_params']['save_folder'] = 'out/D1_sigmoid_avg/thresh_{}/training_d_{}/'.format(sig_thresh, train_d)
                    # print(d_scale, e_scale)

                    for test_name in params['test_suite'].keys():
                        params['test_suite'][test_name]['period'] = 1000


                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()

class sigmoid_D1_avg_slope:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        # algs, n_epochs, lr_list = ['sgd'], [10000], [1e-3]
        algs, n_epochs, lr_list = ['adam'], [4000], [5e-4]

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_d in [True]:#, False]:
                for sig_slope in [50, 5., 1, .1]:#1.]:
                    for sig_thresh in [.1, 1.]:#, .1]:
                        params = deepcopy(default_D1_pars)

                        params['sampler_params']['decays'] = [0.8]
                        params['sampler_params']['batch_size'] = 4096
                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                        params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7, 'loss_name': 'avg_d1'})

                        params['net_params']['activation_type'] = 'Sigmoid'
                        params['net_params']['sigmoid_threshold'] = sig_thresh
                        params['net_params']['sigmoid_slope'] = sig_slope
                        params['net_params']['sigmoid_random_bias'] = False
                        params['net_params']['sigmoid_train_bias'] = False
                        params['net_params']['n'] = 1000
                        params['net_params']['save_folder'] = 'out/D1_sigmoid_avg/slope_{}_thresh_{}/training_d_{}/'.format(sig_slope, sig_thresh, train_d)
                        # print(d_scale, e_scale)

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        params['test_suite']['fit_internal_representation']['batch_size'] = 512
                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()


class randomly_biased_sigmoid_D1_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        # algs, n_epochs, lr_list = ['sgd'], [10000], [1e-3]
        algs, n_epochs, lr_list = ['adam'], [10000], [5e-5]
        train_d = True

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_bias in [False]:
                for sig_slope in [50.]:#1.]:
                    for sig_thresh in [.1,]:#, .1]:
                        params = deepcopy(default_D1_pars)

                        params['sampler_params']['decays'] = [0.8]
                        params['sampler_params']['batch_size'] = 4096
                        params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                        params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-8, 'loss_name': 'avg_d1'})

                        params['net_params']['activation_type'] = 'Sigmoid'
                        params['net_params']['sigmoid_threshold'] = sig_thresh
                        params['net_params']['sigmoid_slope'] = sig_slope
                        params['net_params']['sigmoid_random_bias'] = False
                        params['net_params']['sigmoid_train_bias'] = train_bias
                        n = 1000
                        params['net_params']['n'] = n
                        params['net_params']['save_folder'] = 'out/D1_sigmoid_avg/n_{}_slope_{}_thresh_{}_train_bias_{}/'.format(n, sig_slope, sig_thresh, train_bias)
                        # if params['net_params']['sigmoid_random_bias']:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] + 'randomly_biased/'
                        # if train_bias:
                        #     params['net_params']['save_folder'] = params['net_params']['save_folder'] +
                        # print(d_scale, e_scale)

                        for test_name in params['test_suite'].keys():
                            params['test_suite'][test_name]['period'] = 1000

                        params['test_suite']['fit_internal_representation']['batch_size'] = 512
                        os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                        with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                            json.dump(params, f, indent=4)

                        main(params, seed)
                        gc.collect()


class relu_D1_for_currents_plot:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [8e-5, 2e-1], [4000, 10000], [10, 4], [5, 3]

        # algs, lrs, n_epochs, Ts = ['adam',], [8e-5], [2000,], [5]
        # algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [2.]
        algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [2.]
        # algs, lrs, n_epochs, Ts, Ss = ['adam'], [5e-4], [500], [3], [2.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            # for T in [10, 4]:
            params = deepcopy(default_D1_pars)
            params['sampler_params']['epoch_length'] = T
            params['sampler_params']['scales'] = [s]
            params['sampler_params']['decays'] = [0.995]
            params['net_params']['init_vectors_scales']: [1., 1.] # try to make the prefered scale equal to 2
            params['sampler_params']['batch_size'] = 4096
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
            params['net_params']['save_folder'] = 'out/D1_relu_for_currents_plot/{}/T_{}/'.format(a, T)
            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['n'] = 1000

            params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 5

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()

class relu_D1:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [8e-5, 2e-1], [4000, 10000], [10, 4], [5, 3]

        # algs, lrs, n_epochs, Ts = ['adam',], [8e-5], [2000,], [5]
        # algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [2.]
        algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [1.]
        # algs, lrs, n_epochs, Ts, Ss = ['adam'], [5e-4], [500], [3], [2.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            # for T in [10, 4]:
            params = deepcopy(default_D1_pars)
            params['sampler_params']['epoch_length'] = T
            params['sampler_params']['scales'] = [s]
            params['sampler_params']['decays'] = [0.995]
            params['net_params']['init_vectors_scales']: [1., 1.] # try to make the prefered scale equal to 2
            params['sampler_params']['batch_size'] = 4096
            params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
            params['net_params']['save_folder'] = 'out/D1_relu/{}/T_{}/'.format(a, T)
            params['net_params']['saturations'] = [0, 1e8]
            params['net_params']['n'] = 1000

            params['test_suite'].update({'relu_D1_currents': {'T': 200, 'period': 2**20}})
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = e // 5

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()

class sigmoid_D1_batch:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        algs, n_epochs, lr_list = ['sgd'], [10000], [5e-1]
        # algs, n_epochs, lr_list = ['adam'], [10000], [1e-4]
        s_list = [1.,]

        decay = 0.8
        train_d = True
        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_bias in [False]: #True,
                    params = deepcopy(default_D1_pars)

                    params['sampler_params']['decays'] = [decay]
                    params['sampler_params']['batch_size'] = 2048
                    params['sampler_params']['epoch_length'] = 3
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})#, 'loss_name': 'avg_d1'})
                    # if small_init:
                    #     params['train_params'].update({'train_d_only': True, 'd_init': 'small'})#, 'loss_name': 'avg_d1'})
                    # else:
                    params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7, 'loss_name': 'batch'})

                    params['net_params']['save_folder'] = 'out/D1_sigmoid/training_bias_{}/decay_{}/'.format(train_bias, decay)
                    params['net_params']['activation_type'] = 'Sigmoid'
                    params['net_params']['sigmoid_threshold'] = .1
                    params['net_params']['sigmoid_random_bias'] = False
                    params['net_params']['sigmoid_slope'] = 50.
                    params['net_params']['sigmoid_train_bias'] = False
                    params['net_params']['n'] = 1000
                    # print(d_scale, e_scale)


                    for test_name in params['test_suite'].keys():
                        params['test_suite'][test_name]['period'] = 1000


                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()

class sigmoid_D1_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, n_epochs, Ts = ['sgd'], [30000], [3]
        # Ss = [10., 5., 2., 1., ]
        # lrs = [5e0, 1e0, 1e-1, 1e-1, ]

        # algs, n_epochs, lr_list = ['sgd'], [5000], [1e-2]
        algs, n_epochs, lr_list = ['adam'], [5000], [5e-4]

        for a, e, l in zip(algs, n_epochs, lr_list):
            for train_d in [True]:#, False]:
                for n in [1024, 256, 512, 2048, 4096]:
                    params = deepcopy(default_D1_pars)

                    params['sampler_params']['decays'] = [0.8]
                    params['sampler_params']['batch_size'] = 4096
                    params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e})
                    params['train_params'].update({'train_d_only': train_d, 'stop_loss': 1e-7, 'loss_name': 'avg_d1'})

                    params['net_params']['activation_type'] = 'Sigmoid'
                    params['net_params']['sigmoid_threshold'] = .1
                    params['net_params']['sigmoid_random_bias'] = False
                    params['net_params']['sigmoid_slope'] = 50.
                    params['net_params']['sigmoid_train_bias'] = False
                    params['net_params']['n'] = n
                    params['net_params']['save_folder'] = 'out/D1_sigmoid_avg/n_{}/'.format(n)
                    # print(d_scale, e_scale)

                    for test_name in params['test_suite'].keys():
                        params['test_suite'][test_name]['period'] = 1000

                    params['test_suite']['fit_internal_representation']['batch_size'] = 128
                    params['test_suite'] = {'sanity_check': {'T': 200, 'period': 1000}, 'error_realtime': {'T': 200, 'period': 1000}}
                    os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                    with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    main(params, seed)
                    gc.collect()

class relu_D1_avg:
    def __init__(self):
        pass

    def __call__(self, seed):
        # algs, lrs, n_epochs, Ts, Ss = ['adam', 'sgd'], [8e-5, 2e-1], [4000, 10000], [10, 4], [5, 3]

        # algs, lrs, n_epochs, Ts = ['adam',], [8e-5], [2000,], [5]
        # algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-1], [5000], [3], [2.]
        algs, lrs, n_epochs, Ts, Ss = ['sgd'], [5e-2], [5000], [3], [1.]
        # algs, lrs, n_epochs, Ts, Ss = ['adam'], [5e-4], [500], [3], [2.]


        for a, l, e, T, s in zip(algs, lrs, n_epochs, Ts, Ss):
            for n in [256, 512, 1024, 2048, 4096]:
                params = deepcopy(default_D1_pars)
                params['sampler_params']['epoch_length'] = T
                params['sampler_params']['scales'] = [s]
                params['sampler_params']['decays'] = [0.8]
                params['net_params']['init_vectors_scales']= [1., 1.] # try to make the prefered scale equal to 2
                params['sampler_params']['batch_size'] = 4096
                params['train_params'].update({'optimizer_name': a, 'lr': l, 'n_epochs': e, 'stop_loss':0, 'loss_name': 'avg_d1'})#, 'loss_name': 'avg_d1'})
                params['net_params']['save_folder'] = 'out/D1_relu_avg/n_{}/'.format(n)
                params['net_params']['saturations'] = [0, 1e8]
                params['net_params']['n'] = 1000

                params['test_suite'] = {'sanity_check': {'T': 200, 'period': 1000}, 'error_realtime': {'T': 200, 'period': 1000}}

                os.makedirs(params['net_params']['save_folder'], exist_ok=True)
                with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                    json.dump(params, f, indent=4)

                main(params, seed)
                gc.collect()

if __name__ == '__main__':
    # This part is for efficient multi thread logging
    n_threads = 4
    n_seeds = 8
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = Pool(n_threads, initializer=install_mp_handler)

    ### SGD gets generalizing as soon as T>=3
    ### For T=3, Adam gets much better loss, but poorly generalizes.
    # pool.map(linear_D1_batch_T(), range(n_seeds))
    # pool.map(linear_D1_avg_T(), range(n_seeds))
    # pool.map(linear_D1_cvg_experimental(), range(n_seeds))

    # pool.map(relu_D1_avg(), range(n_seeds))
    # pool.map(relu_D1_batch_T(), range(n_seeds))
    # pool.map(relu_D1_cvg_experimental(), range(n_seeds))
    # pool.map(relu_D1_for_dots_figure(), range(n_seeds))
    # pool.map(relu_D1_for_hessian(), range(n_seeds))

    # Not interesting for now
    # pool.map(linear_D1_avg_train_ed(), range(n_seeds))

    # pool.map(dale_D1_avg(), range(n_seeds))
    # pool.map(dale_D1_batch(), range(n_seeds))
    # pool.map(dale_D1_avg_inhib_frac(), range(n_seeds))
    # pool.map(dale_D1_batch_inhib_frac(), range(n_seeds))
    # pool.map(relu_biased_D1(), range(n_seeds))

    # pool.map(relu_D1_train_d_vary_norms(), range(n_seeds))
    # pool.map(relu_D1_train_d_vary_s(), range(n_seeds))
    # pool.map(relu_D1_proxy_train_d_vary_s(), range(n_seeds))

    # pool.map(sigmoid_D1_avg_threshold(), range(n_seeds))
    # pool.map(sigmoid_D1_avg_slope(), range(n_seeds))
    # pool.map(randomly_biased_sigmoid_D1_avg(), range(n_seeds))

    # pool.map(sigmoid_D1_batch(), range(n_seeds))
    # pool.map(relu_D1_for_currents_plot(), range(n_seeds))
    pool.map(sigmoid_D1_avg(), range(n_seeds))
    # pool.map(relu_D1_avg(), range(n_seeds))
