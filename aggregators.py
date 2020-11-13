# This file to contain all post-processing that aggregates results of several runs into a single figure
import os
import numpy as np
import torch as tch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import orth
from tqdm import tqdm
from shutil import copyfile
import json
import sys
from copy import deepcopy
from scipy.stats import linregress
from nets import ManyChannelsIntegrator


colors = {'purple': '#884ea0', 'blue': '#2471a3', 'green': '#229954',
          'ocre': '#af601a', 'metal': '#717d7e', 'red': '#b03a2e'}

def repeat_in_direct_subfolders(folder, function):
    # this is only useful for one-before-last folders
    for f in os.listdir(folder):
        if not os.path.isfile(folder+f):
            function(folder+f+'/')

def aggregate_loss_across_seeds(folder, n_seeds=8, use_best=True, loglog=False,):
    # Aggregate all loss trajectories into a single, nicer plot
    subfolders = [folder + 'seed{}/'.format(i) for i in range(n_seeds)]
    losses = []
    for path in subfolders:
        try:
            tmp = np.loadtxt(path + 'losses.txt')
            losses.append(np.trim_zeros(tmp, 'b'))
        except:
            print(path + 'losses.txt was not found')
    max_len = 0
    for loss in losses:
        if len(loss) > max_len:
            max_len = len(loss)
    for idx, loss in enumerate(losses):
        if len(loss) < max_len:
            losses[idx] = np.concatenate((loss, loss[-1] * np.ones(max_len-len(loss))), axis=0)
    losses = np.stack(losses, axis=0)

    if use_best:
        bests = losses[:, 0]
        for t in range(1, max_len):
            bests = np.minimum(bests, losses[:, t])
            losses[:, t] = bests
        folder += 'best_'
        np.savetxt(folder + 'losses_agg.txt', losses)

    fig, ax = plt.subplots()
    for i in range(n_seeds):
        ax.semilogy(losses[i])
    if loglog:
        n_epochs = losses.shape[1]
        tmp = np.arange(n_epochs).astype(np.float)**(-2)
        tmp_end = tmp[-1]
        ax.set_xscale('log')
        for i in range(n_seeds):
            end = losses[i, -1]
            ax.semilogy(range(n_epochs), tmp * end / tmp_end, c='k', ls='--')
    ax.set_xlabel(r'Number of optimization steps')
    ax.set_ylabel(r'Value of the Loss')
    fig.savefig(folder+'summary_loss.pdf')

    return losses.shape[1]


def aggregate_condition_linear():
    n_seeds = 20
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
    conds_avgs = np.zeros((len(e_scales), len(s_range)))
    conds_stds = np.zeros((len(e_scales), len(s_range)))

    dots_avg = np.zeros((len(e_scales)))
    dots_log_std = np.zeros((len(e_scales)))

    scales_idx = 0

    for s_e, s_d, o in zip(e_scales, d_scales, overlaps):
        folder = 'out/convergence_D1/linear_hessian/e_{}_d_{}_o_{}/'.format(s_e, s_d, o)
        subfolders = [folder + 'seed{}/'.format(i) for i in range(n_seeds)]
        plop_conds = np.zeros((n_seeds, len(s_range)))
        plop_mins = np.zeros((n_seeds, len(s_range)))
        plop_maxs = np.zeros((n_seeds, len(s_range)))
        dots = np.zeros((n_seeds))

        for idx, f in enumerate(subfolders):
            eigs = np.load(f + 'eigs.npy')
            plop_conds[idx] = np.exp(np.nanmin(np.log(eigs[:, :, -1])-np.log(eigs[:, :, 1]), axis=1))
            dots[idx] = np.loadtxt(f + 'e_dot_d.txt')

        # print(plop_conds.min(), plop_conds.max())

        conds_avgs[scales_idx] = np.exp(np.nanmean(np.log(plop_conds), axis=0, dtype=np.float64))
        conds_stds[scales_idx] = np.nanstd(np.log(plop_conds), axis=0) + np.finfo(np.float64).eps

        dots_avg[scales_idx] = np.exp(np.nanmean(np.log(dots), axis=0))
        dots_log_std[scales_idx] = np.nanstd(np.log(dots), axis=0)
        scales_idx += 1

    params = {'figure.figsize': [6.47, 4.3],
          'axes.labelsize' : 8,
          'font.size' : 8,
          'xtick.labelsize' : 7,
          'ytick.labelsize' : 7,
          'text.usetex': True,
          'text.latex.preamble': r"\usepackage{amsmath},\usepackage{amssymb},\usepackage{bm}",}

    with plt.rc_context(rc=params):
        fig, axes = plt.subplots(2,2)
        for idx, legend in enumerate([r'Changing $\bm{d}$ scale', r'Changing $\bm{e}$ scale', 'Changing overlap', 'Changing both scales']):
            ax = axes[idx //2, idx%2]

            ax.axvline(e_scales[2*idx] * d_scales[2*idx], c=colors['purple'])
            ax.axvline(dots_avg[2*idx], c=colors['purple'], ls ='--')
            ax.axvspan(np.exp(np.log(dots_avg[2*idx])-dots_log_std[2*idx]), np.exp(np.log(dots_avg[2*idx])+dots_log_std[2*idx]), color=colors['purple'], alpha=0.25)
            ax.axvline(e_scales[2*idx+1] * d_scales[2*idx+1], c=colors['ocre'])
            ax.axvline(dots_avg[2*idx+1], c=colors['ocre'], ls ='--')
            ax.axvspan(np.exp(np.log(dots_avg[2*idx+1])-dots_log_std[2*idx+1]), np.exp(np.log(dots_avg[2*idx+1])+dots_log_std[2*idx+1]), color=colors['ocre'], alpha=0.25)

            ax.set_xlabel(r'Scale $s$')
            ax.set_ylabel(r'Condition number $\mathcal{C}$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            print(s_range.shape, conds_avgs[2*idx].shape, conds_avgs[2*idx].min(), conds_avgs[2*idx].max(), np.mean(np.isnan(conds_avgs[2*idx])), np.mean(conds_avgs[2*idx] < 0))
            if idx == 0:
                label1 = r'$|\bm{d}|$=' + '{}'.format(d_scales[2*idx])
                label2 = r'$|\bm{d}|$=' + '{}'.format(d_scales[2*idx+1])
            elif idx == 1:
                label1 =  r'$|\bm{e}|$=' + '{}'.format(e_scales[2*idx])
                label2 =  r'$|\bm{e}|$=' + '{}'.format(e_scales[2*idx+1])

            elif idx == 2:
                label1 = r'$o$=' + '{}'.format(overlaps[2*idx])
                label2 = r'$o$=' + '{}'.format(overlaps[2*idx+1])
            elif idx == 3:
                label1 =  r'$|\bm{e}|$=' + '{}'.format(e_scales[2*idx])
                label1 += r'$, |\bm{d}|$=' + '{}'.format(d_scales[2*idx])
                label2 =  r'$|\bm{e}|$=' + '{}'.format(e_scales[2*idx+1])
                label2 += r'$, |\bm{d}|$=' + '{}'.format(d_scales[2*idx+1])


            ax.plot(s_range, conds_avgs[2*idx], c=colors['purple'], label=label1)
            # print(np.exp(np.log(conds_avgs[2*idx]) + conds_stds[2*idx] - (np.log(conds_avgs[2*idx]) - conds_stds[2*idx])) )
            ax.fill_between(s_range,  np.exp(np.log(conds_avgs[2*idx]) - conds_stds[2*idx]), np.exp(np.log(conds_avgs[2*idx]) + conds_stds[2*idx]), color=colors['purple'], alpha=.5)
            ax.plot(s_range, conds_avgs[2*idx+1], c=colors['ocre'], label=label2)
            ax.fill_between(s_range,  np.exp(np.log(conds_avgs[2*idx+1]) - conds_stds[2*idx+1]), np.exp(np.log(conds_avgs[2*idx+1]) + conds_stds[2*idx+1]), color=colors['ocre'], alpha=.5)
            # ax.fill_between(s_range, np.nanmax(conds[2*idx+1], axis=0), np.nanmin(conds[2*idx+1], axis=0), color=colors['ocre'], alpha=.5)
            if idx !=3:
                ax.legend()
            else:
                ax.legend(loc=3)
            ax.set_title(legend)
            idx += 1

        from matplotlib.lines import Line2D

        l1 = Line2D([0], [0], color='k', ls='--', label=r'$d\cdot e$')
        l2 = Line2D([0], [0], color='k', label=r'$|d||e|$')

        # Create the legend
        fig.legend(handles=[l1, l2],
                   loc="center right",   # Position of legend
                   )

        fig.tight_layout()
        fig.savefig('out/convergence_D1/linear_hessian/summary.pdf')
        plt.close(fig)

def convergence_experimental_relu_D1():
    n_seeds = 8
    folder_template = 'out/D1/relu_cvg_experimental/s_{}/lr_{}/'
    s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
    lr_list = [4e-4, 1e-3, 1e-2, 8e-2, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]
    fit_ranges = [
        [10000, 25000],
        [4000, 12000],
        [450, 1250],
        [150, 300],
        [60, 100],
        [100, 400],
        [200, 1500],
        [1000, 10000],
        [5000, 30000],
    ]

    # Some experiments have missing values of the loss, but they still entered exp cvg and worked (from figure) so we still use them
    fitted_coefs = np.zeros((len(s_list), n_seeds))
    for s_idx, s in enumerate(s_list):
        folder = folder_template.format(s, lr_list[s_idx])
        for seed in range(n_seeds):
            start, stop = fit_ranges[s_idx]
            loss = np.loadtxt(folder+'seed{}/losses.txt'.format(seed))
            if len(loss) < stop:
                stop = len(loss)
            loss = loss[start:stop]
            loss /= loss[0]
            logloss = np.log(loss)
            ts = range(len(loss))
            res = linregress(ts, logloss)
            fitted_coefs[s_idx, seed] = res[0]

    fitted_coefs[fitted_coefs>0] = np.nan
    # The fitted coefs should be -1/tau_opt, and for plot log_tau is better
    # We are very close to eta optimal, so we forget the eta/eta_opt correction
    fitted_log_eta = -np.log(-fitted_coefs)
    log_eta_mean = np.nanmean(fitted_log_eta, axis=1)
    log_eta_std = np.nanstd(fitted_log_eta, axis=1)

    plt.figure()
    plt.plot(s_list, np.exp(log_eta_mean))
    plt.fill_between(s_list, np.exp(log_eta_mean-log_eta_std), np.exp(log_eta_mean+log_eta_std))
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('out/D1/relu_cvg_experimental/summary.pdf')


def final_dots_relu_D1():
    folder_template = 'out/D1/relu_dots/s_{}/'
    s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
    n_seeds = 8

    final_dots = np.zeros((4,len(s_list), n_seeds))
    final_sigmas = np.zeros((len(s_list), n_seeds))
    for s_idx, s in enumerate(s_list):
        folder = folder_template.format(s)
        for seed in range(n_seeds):
            net = tch.load(folder+'seed{}/best_net.pt'.format(seed))
            e = net.encoders[0].detach().cpu().numpy()
            d = net.decoders[0].detach().cpu().numpy()
            l = np.load(folder+'seed{}/final/lefts.npy'.format(seed))[:,0]
            r = np.load(folder+'seed{}/final/rights.npy'.format(seed))[:,0]
            final_dots[0, s_idx, seed] = r.dot(e)
            final_dots[1, s_idx, seed] = r.dot(d)
            final_dots[2, s_idx, seed] = l.dot(d)
            final_sigmas[s_idx, seed] = np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed))[0]

    sigmas_avg = np.mean(np.log(final_sigmas), axis=1)
    sigmas_std = np.std(np.log(final_sigmas), axis=1)

    final_dots_avg = np.nanmean(np.log(final_dots), axis=2)
    final_dots_std = np.nanstd(np.log(final_dots), axis=2)

    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes[0].plot(s_list, np.exp(sigmas_avg))
    axes[0].fill_between(s_list, np.exp(sigmas_avg-sigmas_std), np.exp(sigmas_avg+sigmas_std), alpha=.5)
    tmp = np.linspace(.05, 10, 1000)
    tmp2 = [2*x if x>1 else 2 for x in tmp]
    axes[0].plot(tmp, tmp2, ls='--', c='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'Scale $s$')
    axes[0].set_ylabel(r'Singular value $\sigma$')

    labels = ['r dot e', 'd dot r', 'd dot l']
    styles = ['-', '--', '-.']
    for idx in range(3):
        axes[1].plot(s_list, np.exp(final_dots_avg[idx]), ls=styles[idx], label=labels[idx])
        axes[1].fill_between(s_list, np.exp(final_dots_avg[idx]-final_dots_std[idx]),
                            np.exp(final_dots_avg[idx]+final_dots_std[idx]), alpha=.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'Scale $s$')
    axes[1].set_ylabel('Dot products')

    fig.savefig('out/D1/relu_dots/summary.pdf')

def aggregate_hessian_relu_D1(compute=True):
    folder_template = 'out/D1/relu_hessian/s_{}/'
    s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
    n_seeds = 8
    gamma = .995
    n = 50

    max_eigs = np.zeros((len(s_list), n_seeds))

    for s_idx, s in enumerate(s_list):
        folder = folder_template.format(s)
        for seed in range(n_seeds):
            if compute:
                net = tch.load(folder+'seed{}/best_net.pt'.format(seed))

                d = net.encoders[0].detach().cpu().numpy()
                e = net.encoders[0].detach().cpu().numpy()
                l = np.load(folder+'seed{}/final/lefts.npy'.format(seed))[:,0]
                r = np.load(folder+'seed{}/final/rights.npy'.format(seed))[:,0]
                sig = np.loadtxt(folder + 'seed{}/final/sigmas.txt'.format(seed))[0]
                del net
                W = sig * np.outer(l, r)

                We = np.dot(W, e)
                W2_e = np.dot(W, We)
                H_p_We = np.dot(W, e) > 0
                H_m_We = np.dot(W, e) < 0
                r_p_We = np.dot(W, e) * H_p_We
                r_m_We = - np.dot(W, e) * H_m_We
                d_H_p = d * H_p_We
                d_H_m = d * H_m_We

                # Here, use W and avoid l r as much as possible so we could reuse it eg for Newton method
                # H will be a matrix in the end, for factorizable terms it's very easy to write
                hessian = np.zeros((n**2, n**2))

                hessian += np.outer(np.outer(d_H_p, e).flatten(), np.outer(d_H_p, e).flatten())
                hessian += np.outer(np.outer(d_H_m, e).flatten(), np.outer(d_H_m, e).flatten())
                for a in range(n):
                    hessian += np.outer(np.outer(W[a] * H_p_We, e).flatten(), np.outer(W[a] * H_p_We, e).flatten())
                    hessian += np.outer(np.outer(W[a] * H_m_We, e).flatten(), np.outer(W[a] * H_m_We, e).flatten())


                # For non-factorizable parts, use 4D tensor and iterate over it
                # Complexity is power 4 because of the last term
                tensor_hessian = np.zeros((n,n,n,n))

                # Term with a delta_ik
                for i in tqdm(range(n)):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                if i == k:
                                    tensor_hessian[i, j, i, l] += (r_p_We[j]-gamma*e[j])*(r_p_We[l]-gamma*e[l])
                                    tensor_hessian[i, j, i, l] += (r_m_We[j]+gamma*e[j])*(r_m_We[l]+gamma*e[l])
                                if j == k:
                                    tensor_hessian[i, j, j, l] += (W2_e[i]-2*gamma*We[i])*e[l]
                                if i == l:
                                    tensor_hessian[i, j, j, l] += (W2_e[i]-2*gamma*We[i])*e[l]

                                tensor_hessian[i, j, k, l] += (We[j]-2*gamma*e[j])*W[i, k]*e[l]
                                tensor_hessian[i, j, k, l] += (We[l]-2*gamma*e[l])*W[k, i]*e[j]

                tensor_hessian = tensor_hessian.reshape((n**2,n,n))
                tensor_hessian = tensor_hessian.reshape((n**2,n**2))
                hessian += tensor_hessian
                # Hessian is symmetric so use it !
                eigs, _ = np.linalg.eigh(hessian)
                np.savetxt(folder + 'seed{}/final_hessian_eigs.txt'.format(seed), eigs)
                max_eigs[s_idx, seed] = eigs.max()

                fig, axes = plt.subplots(2)
                axes[0].hist(np.log10(eigs[np.where(eigs>0)]), density=False, bins=200)
                axes[0].set_title('Histogram of log10 of positive eigs')
                axes[1].hist(np.log10(-eigs[np.where(eigs<0)]), density=False, bins=200)
                axes[1].set_title('Histogram of log10 of norm of negative eigs')
                fig.tight_layout()
                fig.savefig(folder + 'seed{}/final_hessian_eigs.pdf'.format(seed))
                plt.close()

            else:
                max_eigs[s_idx, seed] = np.loadtxt(folder + 'seed{}/final_hessian_eigs.txt'.format(seed)).max()


    log_mean = np.mean(np.log(max_eigs), axis=-1)-2*np.log(s_list)# Because ze used loss over s**2
    log_std = np.std(np.log(max_eigs), axis=-1)
    plt.figure()
    plt.plot(s_list, np.exp(log_mean))
    plt.fill_between(s_list, np.exp(log_mean-log_std), np.exp(log_mean+log_std))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Scale $s$')
    plt.ylabel('Maximum eigenvalue')
    plt.savefig('out/D1/relu_hessian/summary.pdf')


def aggregate_dale():
    folder = 'out/D2/dale/inhib_frac_0.25/'
    n_seeds = 16
    n = 1000
    exc_idx = (0, 750)
    inh_idx = (750, 1000)


    lefts_elements = np.zeros((n_seeds, 3, n))
    rights_elements = np.zeros((n_seeds, 3, n))


    for seed in range(n_seeds):
        net = tch.load(folder+'seed{}/best_net.pt'.format(seed))
        W = net.W.mm(tch.diag(net.synapse_signs))
        U, sigmas, V = tch.svd(W, compute_uv=True)
        lefts_elements[seed] = U[:,:3].transpose(0,1).detach().cpu().numpy()
        rights_elements[seed] = V[:,:3].transpose(0,1).detach().cpu().numpy()
        del U, V

    lefts_exc = lefts_elements[:,:, range(*exc_idx)]
    lefts_inh = lefts_elements[:,:, range(*inh_idx)]

    rights_exc = rights_elements[:,:, range(*exc_idx)]
    rights_inh = rights_elements[:,:, range(*inh_idx)]

    dale_left_exc = lefts_exc[:, 0, :].flatten()
    dale_left_inh = lefts_inh[:, 0, :].flatten()
    dale_right_exc = rights_exc[:, 0, :].flatten()
    dale_right_inh = rights_inh[:, 0, :].flatten()

    other_left_exc = lefts_exc[:, 1:, :].flatten()
    other_left_inh = lefts_inh[:, 1:, :].flatten()
    other_right_exc = rights_exc[:, 1:, :].flatten()
    other_right_inh = rights_inh[:, 1:, :].flatten()


    fig, axes = plt.subplots(2)
    sns.distplot(dale_left_exc, ax=axes[0], color='red', kde=False, norm_hist=True, label='Excitatory neurons')
    sns.distplot(dale_left_inh, ax=axes[0], color='blue', kde=False, norm_hist=True, label='Inhibitory neurons')
    sns.distplot(dale_right_exc, ax=axes[1], color='red', kde=False, norm_hist=True, label='Excitatory neurons')
    sns.distplot(dale_right_inh, ax=axes[1], color='blue', kde=False, norm_hist=True, label='Inhibitory neurons')
    axes[0].set_xlabel('Coefficient of left singular vector')
    axes[1].set_xlabel('Coefficient of right singular vector')
    axes[0].set_ylabel('Probability density')
    axes[1].set_ylabel('Probability density')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig('out/D2/dale/inhib_frac_0.25/dale_mode_singular_vectors.pdf')

    fig, axes = plt.subplots(2)
    sns.distplot(other_left_exc, ax=axes[0], color='red', kde=False, norm_hist=True, label='Excitatory neurons')
    sns.distplot(other_left_inh, ax=axes[0], color='blue', kde=False, norm_hist=True, label='Inhibitory neurons')
    sns.distplot(other_right_exc, ax=axes[1], color='red', kde=False, norm_hist=True, label='Excitatory neurons')
    sns.distplot(other_right_inh, ax=axes[1], color='blue', kde=False, norm_hist=True, label='Inhibitory neurons')
    axes[1].set_xlabel('Coefficient of right singular vector')
    axes[0].set_xlabel('Coefficient of left singular vector')
    axes[1].set_xlabel('Coefficient of right singular vector')
    axes[0].set_ylabel('Probability density')
    axes[1].set_ylabel('Probability density')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig('out/D2/dale/inhib_frac_0.25/other_modes_singular_vectors.pdf')

def aggregate_sigmas_D3():
    folders = ['out/D3/relu/adam/T_4/', 'out/D3_sigmoid_batch/n_1000_slope_50.0_thresh_0.1_train_bias_False/']
    n_seeds = 4
    n = 1000
    # sigmas = np.zeros((n_seeds, n))
    for folder in folders:
        sigmas = []
        plt.figure()
        for seed in range(n_seeds):
            print(seed)
            # print(np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed)).shape)
            # sigmas[seed] = np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed))
            sigmas.append(np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed)))
            # plt.hist(np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed)), bins=20, log=True, histtype='step', fill=True)
        plt.hist(sigmas, bins=20, log=True, histtype='bar', fill=True, color=['#85c1e9', '#27ae60', '#d68910', '#c39bd3'])
        plt.savefig(folder+'aggregated_sigmas.pdf')
        plt.close()


def aggregate_angles_D2(folder='out/D2/relu/adam/T_10/'):
    n_seeds = 16
    n = 1000
    # sigmas = np.zeros((n_seeds, n))
    thetas = []

    plt.figure()
    for seed in range(n_seeds):
        print(seed)
        # print(np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed)).shape)
        # sigmas[seed] = np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed))
        selectivity_vectors = np.load(folder+'seed{}/final/selectivity_vectors.npy'.format(seed))
        angles = np.arctan2(selectivity_vectors[:,1], selectivity_vectors[:,0])
        thetas.append(angles)

        # plt.hist(np.loadtxt(folder+'seed{}/final/sigmas.txt'.format(seed)), bins=20, log=True, histtype='step', fill=True)

    thetas = np.stack(thetas, axis=0)
    min, max  = np.min(thetas), np.max(thetas)
    hists = []

    bins = 20
    for seed in range(n_seeds):
        if seed == 0:
            y, binEdges = np.histogram(thetas[seed], range=(min,max), bins=bins)
            hists.append(y)
        else:
            hists.append(np.histogram(thetas[seed], range=(min,max), bins=bins)[0])

    hists = np.stack(hists, axis=0)
    mean, std = np.mean(hists, axis=0), np.std(hists, axis=0)

    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    width      = 5 / bins

    plt.bar(bincenters, mean, width=width, color='r', yerr=std)
    plt.show()
    plt.savefig(folder+'aggregated_sigmas.pdf')
    plt.close()


def aggregate_fit_coefs_relu(main_folder, n_bins=10, n=1000, n_seeds=10):
    subfolders = [main_folder + 'seed{}/'.format(i) for i in range(n_seeds)]
    c_pos_agg = np.zeros((n, n_seeds))
    c_neg_agg = np.zeros((n, n_seeds))
    is_pos_agg = np.zeros((n, n_seeds))
    is_neg_agg = np.zeros((n, n_seeds))
    is_sha_agg = np.zeros((n, n_seeds))
    is_nul_agg = np.zeros((n, n_seeds))

    for idx, folder in enumerate(subfolders):
        c_pos_agg[:, idx] = np.loadtxt(folder + 'final/c_pos.txt')
        c_neg_agg[:, idx] = np.loadtxt(folder + 'final/c_neg.txt')
        is_pos_agg[:, idx] = np.loadtxt(folder + 'final/positives.txt')
        is_neg_agg[:, idx] = np.loadtxt(folder + 'final/negatives.txt')
        is_sha_agg[:, idx] = np.loadtxt(folder + 'final/shared.txt')
        is_nul_agg[:, idx] = np.loadtxt(folder + 'final/nulls.txt')

    pop_idx_agg = np.zeros((n, n_seeds))
    pop_idx_agg[np.where(is_pos_agg==1)]=0
    pop_idx_agg[np.where(is_neg_agg==1)]=1
    pop_idx_agg[np.where(is_sha_agg==1)]=2
    pop_idx_agg[np.where(is_nul_agg==1)]=3


    # print(np.sum(pop_idx_agg==2, axis=0))

    xlim = max(c_pos_agg.max(), np.abs(c_neg_agg).max())
    bins = np.linspace(0, xlim, n_bins)
    bin_centers = bins[:-1] + bins[1:]
    bin_width = bins[1]-bins[0]
    labels = ['Positive', 'Negative', 'Shared', 'Null']
    colors = ['r', 'b', 'g', 'gray']

    hists_pos = np.zeros((4, n_bins-1, n_seeds))
    hists_neg = np.zeros((4, n_bins-1, n_seeds))
    for seed in range(n_seeds):
        for pop_idx in range(4):
            tmp, _, _ = plt.hist(c_pos_agg[:, seed][np.where(pop_idx_agg[:, seed]==pop_idx)], bins, density=False, log=True)
            hists_pos[pop_idx, :, seed] = tmp
            tmp, _, _ = plt.hist(c_neg_agg[:, seed][np.where(pop_idx_agg[:, seed]==pop_idx)], bins, density=False, log=True)
            hists_neg[pop_idx, :, seed] = tmp

    plt.close('all')
    mean_pos = np.mean(hists_pos, axis=2)
    mean_neg = np.mean(hists_neg, axis=2)
    std_pos = np.std(hists_pos, axis=2)
    std_neg = np.std(hists_neg, axis=2)

    # Now for the real plot; Do the base histogram as before, then add
    fig, axes = plt.subplots(2, 1, sharex=True)
    c_pos = c_pos_agg.flatten()
    c_neg = c_neg_agg.flatten()
    pop_indices = pop_idx_agg.flatten()

    x_pos = [c_pos[np.where(pop_indices==k)] for k in range(4)]
    x_neg = [-c_neg[np.where(pop_indices==k)] for k in range(4)]

    for idx, plop in enumerate(c_pos):
        c_pos[idx] = plop[~np.isnan(plop)]
    for idx, plop in enumerate(c_neg):
        c_neg[idx] = plop[~np.isnan(plop)]

    axes[0].hist(x_pos, bins, color=colors, alpha=0.5, label=labels, density=False, log=True)
    # for cat in range(4):
    #     axes[0].errorbar(bins[:-1]+(2*cat+1)*.125*bin_width, mean_pos[cat], yerr=std_pos[cat], c=colors[cat])
    axes[1].hist(x_neg, bins, color=colors, alpha=0.5, label=labels, density=False, log=True)
    # axes[1].errorbar(bin_centers, mean_neg, yerr=std_neg)

    fig.tight_layout()
    fig.savefig(main_folder + 'aggregated_histogram_coefs.pdf')
    plt.close('all')


def aggregate_sigmoid_activity_D1(main_folder, n_bins=10, n=1000, n_seeds=8):
    subfolders = [main_folder + 'seed{}/'.format(i) for i in range(n_seeds)]
    agg = []
    decs = []
    mean_acts = []

    for folder_name in subfolders:
        agg.append(np.load(folder_name + 'final/agg_for_activity_hists_sigmoid.npy'))
        try:
            decs.append(tch.load(folder_name + 'best_net.pt').decoders[0].detach().cpu().numpy())
            mean_acts.append(np.load(folder_name + 'final/agg_for_activity_hists_sigmoid.npy').mean(axis=0))
        except:
            pass

    agg = np.stack(agg)



    # Plot for mean activity as a function of decoder (or the other way around)
    # for i, d, m in zip(range(len(decs)), decs, mean_acts):
    #     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #
    #     ax.scatter(np.abs(d).flatten(), m.flatten(), rasterized=True)
    #
    #     ax.legend()
    #     ax.set_aspect('auto')
    #     ax.set_ylabel(r'Mean neuron activation')
    #     ax.set_xlabel(r'Value of decoder')
    #     ax.set_xscale('log')
    #     fig.savefig(main_folder + 'mean_act_vs_d_{}.pdf'.format(i))
    #     plt.close(fig)


    # Activity histogram at different values of y
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    seismic = plt.get_cmap('seismic')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # ax.scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(agg.flatten())), s=4, rasterized=True)
    # for idx, y_bin_idx, y_label, c in zip([0,1,2], [50, 75, 99], ['y=0', 'y=2', 'y=4'], [[0., 0., 1., .3], [0.,1.,0., .3], [1.,0.,0., .3]]):
    for idx, y_bin_idx, y_label, c in zip([0,1,2,3], [50, 62, 75,99], ['y=0', 'y=1', 'y=2', 'y=4'], [[0., 0., 1., .5], [0.,.5,0., .5], [.5,.5,0., .5], [1.,0.,0., .5]]):
        acts = agg[:, y_bin_idx]
        print(acts.min(), acts.max())
        hists = []
        for seed in range(n_seeds):
            hists.append(np.histogram(acts[seed], bins=bins)[0])
            print(hists[-1].shape)

        hists = np.stack(hists)
        print(hists.shape, bins[:-1])

        # yerr_low = np.exp(np.log(hists).mean(axis=0)- np.log(hists).std(axis=0))
        # yerr_high = np.exp(np.log(hists).mean(axis=0)+ np.log(hists).std(axis=0))
        # yerr = np.stack([yerr_low, yerr_high])
        ax.bar(bins[:-1] + idx*.05/3, hists.mean(axis=0), width=.05/3, label=y_label, color=c)
        # ax.errorbar(bins[:-1] + idx*.05/3, hists.mean(axis=0), yerr=yerr, ls='', marker='', )
        ax.set_yscale('log')


    # Looks kinda bad because we always plot the x>0 part of the line, which might be in the inactive plane.
    ax.legend()
    ax.set_aspect('auto')
    ax.set_xlabel(r'Mean neuron activation')
    ax.set_ylabel(r'Index of neuron')
    ax.set_title('Value of activity')
    fig.savefig(main_folder + 'activity_histogram_averaged.pdf')
    plt.close(fig)



def aggregate_distance_to_manifold():
    named_folders = {
    'relu_D1_avg': 'canned_out/D1/relu_avg/adam/',
    'relu_D2_avg': 'canned_out/D2/relu_avg/',
    'relu_D1_batch': 'canned_out/D1_relu/sgd/T_3/',
    'relu_D2_batch': 'canned_out/D2/relu/sgd/T_3/',
    'relu_D3_batch': 'canned_out/D3/relu/adam/T_4/',
    'sigmoid_D1_avg': 'canned_out/D1_sigmoid_avg/',
    'sigmoid_D2_avg': 'canned_out/D2_sigmoid_avg/n_1000_slope_50.0_thresh_0.1_train_bias_False/',
    'sigmoid_D1_batch': 'canned_out/D1_sigmoid/training_bias_False/decay_0.8/',
    'sigmoid_D2_batch': 'canned_out/D2_sigmoid_batch/n_1000_slope_50.0_thresh_0.1_train_bias_True/',
    'sigmoid_D3_batch': 'canned_out/D3_sigmoid_batch/n_1000_slope_50.0_thresh_0.1_train_bias_True/',
    # 'relu_D3_avg': 'out/D3_relu_avg/n_1024/',
    'sigmoid_D3_avg': 'out/D3_sigmoid_avg/n_1024/',
    'relu_D5_avg': 'out/D5_relu_avg/n_1024/',
    'sigmoid_D5_avg': 'out/D5_sigmoid_avg/n_1024/',

    }

    for name, folder in named_folders.items():
        ratios = []
        if name not in ['relu_D5_avg', 'sigmoid_D5_avg', 'relu_D3_avg', 'sigmoid_D3_avg']:
            n_seeds = 8
        else:
            n_seeds = 4
        for seed in range(n_seeds):
            tmp = np.loadtxt(folder+'seed{}/final/representation_plots/norms_current_fit.txt'.format(seed))
            ratios.append(tmp[1]/tmp[0])
        mean, std = np.mean(ratios), np.std(ratios)
        print('For exp {}, found r={:.3e} pm {:.3e}'.format(name, mean, std))

def aggregate_error_n_t():
    folder = 'out/D1_sigmoid_avg/'
    seeds = range(8)
    ns = [256, 512, 1024, 2048, 4096]

    all_values = []
    for n in ns:
        plop = []
        for seed in seeds:
            file = folder+'n_{}/seed{}/final/error_realtime/datablob.txt'.format(n, seed)
            print(file)
            plop.append(np.loadtxt(file).flatten())
        all_values.append(np.concatenate(plop))

    #all_values: n_ns, n_trajs*T*n_seeds all errors for different models/trajs

    all_logs = [.5 * np.log10(data) for data in all_values] #.5 because this is squared error;
    print(all_logs[0], all_logs[-1])
    for idx, n in enumerate(ns):
        sns.distplot(all_logs[idx], label='n={}'.format(n))
    plt.legend()
    plt.savefig(folder+'n_error_study.pdf')

    log_mean_error = [np.log10(np.sqrt(np.mean(data))) for data in all_values]
    # log_mean_error = [np.mean(np.log10(np.sqrt(data))) for data in all_values]
    plt.figure()
    plt.scatter(np.log10(ns), log_mean_error)
    plt.ylabel('Log of the root mean square error')
    plt.xlabel('Log n')
    plt.savefig(folder+'n_error_study_log_mean.pdf')

    # log_mean_error = [np.log10(np.sqrt(np.mean(data))) for data in all_values]
    log_mean_error = [np.mean(np.log10(np.sqrt(data))) for data in all_values]
    plt.figure()
    plt.scatter(np.log10(ns), log_mean_error)
    plt.ylabel('Log of the root mean square error')
    plt.xlabel('Log n')
    plt.savefig(folder+'n_error_study_mean_log.pdf')

if __name__ == '__main__':
    ## This is for comparison between exponential and algebraic convergence
    # aggregate_loss_across_seeds('out/convergence_D1/dot_scale/gaussian_init/')
    # aggregate_loss_across_seeds('out/convergence_D1/dot_scale/null_init/', loglog=True)

    ## This is for the condition number as a function of s plot
    # aggregate_condition_linear()

    ## This one for clearer view in influence of lr at different values of s
    ## Linear case:

    # lr_lists = [
    #     [5e-1, 1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3], # s=0.1, (drop is between 2e-2 and 1e-2)
    #     [1e0, 7e-1, 6e-1, 5e-1, 1e-1], # s=1, (drop is between .6 and .7)
    #     [5e-1, 1e-1, 7e-2, 9e-2, 8e-2, 5e-2], # s=10 (drop is between 8e-2 and 9e-2)
    # ]
    # for s, lr_list in zip([.1, 1., 10], lr_lists):
    #     for lr in lr_list:
    #         aggregate_loss_across_seeds('out/D1/linear_cvg_experimental/s_{}/lr_{}/'.format(s, lr), use_best=False)

    ## ReLU case:
    # s_list = [.05, .1, .3, .6, 1, 2, 4, 10, 20]
    # lr_lists = [
    #     [6e-4, 4e-4], # s=.05, (drop is around 6e-4)
    #     [3e-3, 1e-3], # s=0.1, (drop is around 2e-3)
    #     [3e-2, 1e-2], # s=0.3, (drop is around 2e-2)
    #     [8e-2, 3e-2], # s=0.6, (drop is around 8e-2)
    #     [3e-1, 1e-1], # s=1, (drop is around 2e-1)
    #     [3e-1, 1e-1], # s=2, (drop is around 2e-1)
    #     [3e-1, 1e-1], # s=4, (drop is around 2e-1)
    #     [3e-1, 1e-1], # s=10, (drop is around 2e-1)
    #     [3e-1, 1e-1], # s=20, (drop is around 2e-1)
    # ]
    # for s, lr_list in zip(s_list, lr_lists):
    #     for lr in lr_list:
    #         aggregate_loss_across_seeds('out/D1/relu_cvg_experimental/s_{}/lr_{}/'.format(s, lr), use_best=False)

    # convergence_experimental_relu_D1()
    # final_dots_relu_D1()
    # aggregate_hessian_relu_D1(compute=True)
    # aggregate_dale()
    # aggregate_sigmas_D3()
    # aggregate_angles_D2(folder='out/D2/relu/adam/T_10/')

    # aggregate_sigmoid_activity_D1('out/D1_sigmoid_avg/n_1000_slope_50.0_thresh_0.1_train_bias_False/')
    # aggregate_distance_to_manifold()
    # aggregate_error_n_t()
    aggregate_angles_D2(folder='out/D2_sigmoid_avg/n_1024/')
    aggregate_angles_D2(folder='out/D2_relu_avg/n_1024/')
