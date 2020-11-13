import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from train import main
from copy import deepcopy
import json
import os
import gc
import logging
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool
import torch as tch
from nets import TwoTwoNet

from utils import min, max, sqrtm
from scipy.special import logsumexp
from scipy.linalg import eigvalsh
from losses import average_loss_D1
stable_add = lambda log_abs, signs: logsumexp(a=log_abs, b=signs, return_sign=True)

default_D1_pars = {
    'net_type': 'TwoTwoNet',
    'net_params': {
                    'device_name': 'cuda',
                    'n': 1000,
                    'n_channels': 1,
                    'saturations': [-1e8, 1e8],
                    'init_radius': 0.,
                    'save_folder': None,
                    'init_vectors_type': 'random',
                    'init_vectors_overlap': None,
                    'init_vectors_scales': [1,1],
                    },
    'sampler_params': {
                    'n_channels': 1,
                    'epoch_length': 3,
                    'decays': [.995, ],
                    'scales': [1., ],
                    'batch_size': 1024,
                    },
    'train_params': {
                    'loss_name': 'avg_d1',
                    'optimizer_name': 'sgd',
                    'lr': 1e-1,
                    'n_epochs': 100000,
                    'stop_loss': 1e-9,
                    },
    'test_suite': {
                    'sanity_check': {'T': 200, 'period': 2**20},
                  },
    'rescale_s_dot': False,
    'rescale_s_norms': False,
}

# First, illustrate algebraic vs exponential convergence at dot scale
class cvg_linear_D1_dot_scale:
    def __init__(self):
        pass

    def __call__(self, seed):

        for r, r_name, lr, n_epochs in zip([.1, 0.], ['gaussian_init', 'null_init'], [5e-3, 1e-3], [30000, 100000]):
            params = deepcopy(default_D1_pars)
            params['rescale_s_dot'] = True
            params['net_params']['save_folder'] = 'out/convergence_D1/dot_scale/{}/'.format(r_name)
            params['net_params']['init_vectors_overlap'] = .4
            params['net_params']['init_radius'] = r
            params['train_params']['lr'] = lr
            params['train_params']['n_epochs'] = n_epochs
            for test_name in params['test_suite'].keys():
                params['test_suite'][test_name]['period'] = n_epochs // 10

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            main(params, seed)
            gc.collect()


# Now for the theoretical study
def logdot(a, b, sign_a, sign_b):
    b_was_vector = False
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
        sign_b = sign_b.reshape(-1, 1)
        b_was_vector = True
    elif len(b.shape) > 2:
        raise RuntimeError('logdot does not accept more than 2D arrays')

    max_a = np.max(a,1,keepdims=True)
    max_b = np.max(b,0,keepdims=True)
    exp_a, exp_b = a - max_a, b - max_b
    np.exp(exp_a, out=exp_a)
    np.exp(exp_b, out=exp_b)
    c = np.dot(sign_a*exp_a, sign_b*exp_b)
    sgn_c = np.sign(c)
    log_c = np.log(np.abs(c))
    log_c += max_a + max_b
    if b_was_vector:
        log_c = log_c.flatten()
        sgn_c = sgn_c.flatten()
    return log_c, sgn_c

def isoscale_coordinates(net, s, a_range):
    b_range = np.zeros_like(a_range)
    num_range = net.Z * (a_range-net.a_0)*net.b_0 - s * a_range
    denom_range = net.Z * (a_range-net.a_0) - s
    signs = np.sign(num_range) * np.sign(denom_range)
    lognums = np.log(np.abs(num_range))
    logdens = np.log(np.abs(denom_range))
    for idx, a in enumerate(a_range):
        b_range[idx] = signs[idx] * np.exp(lognums[idx]-logdens[idx])
    return zip(a_range, b_range)

def get_log_w(a, b):
    tmp = np.array([[b, -1], [a*b, -a]])
    log_w = np.log(np.abs(.995 * tmp)) # decay hardcoded here for simplicity
    sgn_w = np.sign(tmp)
    log_w -= np.log(np.abs(b-a))
    sgn_w *= np.sign(b-a)
    return log_w, sgn_w

def get_w(a, b):
    log_w, sgn_w = get_log_w(a, b)
    return sgn_w * np.exp(log_w)

def compute_H_spectrum(net, a, b, s):
    # This is a reimplementation of diag to work with log matrices for improved stability
    log_sqsig, sgn_sqsig = np.log(np.abs(net.sqrt_sigma)), np.sign(net.sqrt_sigma)
    log_left_prod, log_right_prod = [None for _ in range(3)], [None for _ in range(3)]
    sgn_left_prod, sgn_right_prod = [None for _ in range(3)], [None for _ in range(3)]

    # m = 0 done separately to avoid taking log of identity...
    log_left_prod[0], sgn_left_prod[0] = log_sqsig, sgn_sqsig
    log_right_prod[0], sgn_right_prod[0] = log_sqsig, sgn_sqsig

    log_M_m, sgn_M_m = get_log_w(a, b)
    log_M, sgn_M = get_log_w(a, b)

    for m in range(1, 3):
        log_left_prod[m], sgn_left_prod[m] = logdot(log_sqsig, log_M_m, sgn_sqsig, sgn_M_m)
        log_right_prod[m], sgn_right_prod[m] = logdot(log_M_m, log_sqsig, sgn_M_m, sgn_sqsig)
        log_M_m, sgn_M_m = logdot(log_M_m, log_M, sgn_M_m, sgn_M)

    log_H, sgn_H = np.log(1e-36) * np.ones((4,4)), np.zeros((4,4))

    for k in range(1, 3+1):
        log_Qk, sgn_Qk = np.log(1e-36) * np.ones((2,2)), np.zeros((2,2))
        for l in range(k):
            for i in range(2):
                for j in range(2):
                    log_tmp = log_left_prod[l][0, i] + log_right_prod[k-1-l][j, 1]
                    sgn_tmp = sgn_left_prod[l][0, i] * sgn_right_prod[k-1-l][j, 1]
                    log_Qk[i, j], sgn_Qk[i, j] = stable_add([log_Qk[i, j], log_tmp], [sgn_Qk[i, j], sgn_tmp])

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        log_plop = np.log(3+1-k) + log_Qk[a, b] + log_Qk[c, d]
                        sgn_plop = sgn_Qk[a, b] * sgn_Qk[c, d]
                        log_H[2*a+b, 2*c+d], sgn_H[2*a+b, 2*c+d] = stable_add(
                                [log_H[2*a+b, 2*c+d], log_plop],
                                [sgn_H[2*a+b, 2*c+d], sgn_plop])

    # Compute H eigs
    H = sgn_H * np.exp(log_H - 2*np.log(s) - 2*np.log(3))
    # The s/T are here because in practice we normalize the loss, hence the eigs too
    return np.sort(eigvalsh(H))



def get_interesting_quantities_along_isoscale(net, s, a_range):
    coordinates = isoscale_coordinates(net, s, a_range)
    losses = np.zeros(len(a_range))
    eigs = np.zeros((len(a_range), 4))
    i = 0
    for a, b in coordinates:
        net.w.data = tch.from_numpy(get_w(a, b)).to(net.device)
        losses[i] = average_loss_D1(net, **{'epoch_length': 3, 'decays': [.995], 'scales': [s]})
        eigs[i] = compute_H_spectrum(net, a, b, s)
        i += 1
    return losses, eigs




class condition_number():
    def __init__(self):
        pass

    def __call__(self, seed):
        e_scales = [
                   1., 1.,  # Changing d_scale
                   1., .3,  # Changing e_scale
                   1., 1.,  # Changing overlap
                   .5, 2.,  # Changing both_scales
                   ]

        d_scales = [
                    1., .3,  # Changing d_scale
                    1., 1.,  # Changing e_scale
                    1., 1.,  # Changing overlap
                    2., .5, # Changing both_scales
                  ]

        overlaps = [
                    .1, .1,  # Changing d_scale
                    .1, .1,  # Changing e_scale
                    0., .8,  # Changing overlap
                    .1, .1, # Changing both_scales
                  ]

        s_range = np.exp(np.linspace(np.log(1e-3), np.log(1e1), 50)).astype(np.float64)
        a_range = np.linspace(-25, 25, 50).astype(np.float64)


        for s_e, s_d, o in zip(e_scales, d_scales, overlaps):
            params = deepcopy(default_D1_pars)
            params['rescale_s_dot'] = True
            params['net_params']['save_folder'] = 'out/convergence_D1/linear_hessian/e_{}_d_{}_o_{}/seed{}/'.format(s_e, s_d, o, seed)
            # params['net_params']['init_vectors_scales'] = [1., e_scale]
            params['net_params']['init_vectors_overlap'] = o
            params['net_params']['init_vectors_scales'] = [s_d, s_e]
            params['net_params']['n'] = 4000

            os.makedirs(params['net_params']['save_folder'], exist_ok=True)
            with open(params['net_params']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            tch.manual_seed(seed)
            tch.cuda.manual_seed(seed)
            np.random.seed(seed)

            net = TwoTwoNet(params['net_params'])
            np.savetxt(params['net_params']['save_folder']+'e_dot_d.txt', np.array([net.e_dot_d]))

            losses, eigs = np.zeros((len(s_range), len(a_range))), np.zeros((len(s_range), len(a_range), 4))
            for s_idx, s in tqdm(enumerate(s_range)):
                losses[s_idx], eigs[s_idx] = get_interesting_quantities_along_isoscale(net, s, a_range)

                # There might be some points where error is bigger due to numerical issues...
                plt.figure()
                plt.hist(np.log(losses[s_idx], where=np.logical_and(losses[s_idx]!=0, np.isfinite(losses[s_idx]))))
                plt.savefig(net.save_folder + 'sanity_loss_{}.png'.format(s))

                plt.figure()
                plt.semilogy(a_range, eigs[s_idx, :, 3]/eigs[s_idx, :, 1])
                plt.savefig(net.save_folder + 'sanity_cond_s_{}.png'.format(s))

            np.save(net.save_folder + 'losses.npy', losses)
            np.save(net.save_folder + 'eigs.npy', eigs)
            gc.collect()


if __name__ == '__main__':
    if __name__ == '__main__':
        ## This part is for efficient multi thread logging
        # n_threads = 8
        # n_seeds = 8
        # logging.basicConfig(level=logging.INFO)
        # install_mp_handler()
        # pool = Pool(n_threads, initializer=install_mp_handler)
        # pool.map(cvg_linear_D1_dot_scale(), range(n_seeds))

        n_threads = 10
        n_seeds = 20
        logging.basicConfig(level=logging.INFO)
        install_mp_handler()
        pool = Pool(n_threads, initializer=install_mp_handler)
        pool.map(condition_number(), range(n_seeds))
