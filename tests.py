import torch as tch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datagen import sample_data
import os
import logging
import gc
import utils
from scipy import stats

max = lambda x, y: x if x > y else y
min = lambda x, y: x if x < y else y
from utils import lstsq

# Check that our network is a good integrator for long sequences
def sanity_check(net, pars, epoch):
    T = pars['T']
    decays = pars['decays']
    scales = pars['scales']

    sanity_sampler_params = {
    'n_channels': net.n_channels,
    'epoch_length': T,
    'decays': decays,
    'scales': scales,
    'batch_size': 10,
    'mode': 'test',
    'is_switch': net.is_switch,
    }

    fig, axeslist = plt.subplots(net.n_channels_out, 5, figsize=(5*8, net.n_channels*4))
    X, y = sample_data(**sanity_sampler_params)

    preds = net.integrate(X)
    # logging.info(len(preds))
    # logging.info(preds[0].shape)
    if len(preds)>1:
        for traj_index in range(5):
            for c in range(net.n_channels_out):
                axeslist[c, traj_index].plot(preds[c][traj_index].detach().cpu().numpy(), c='r', label='real (channel {})'.format(c))
                axeslist[c, traj_index].plot(y[c][traj_index], c='b', label='expected (channel {})'.format(c))
                axeslist[c, traj_index].legend()
    else:
        for traj_index in range(5):
                axeslist[traj_index].plot(preds[0][traj_index].detach().cpu().numpy(), c='r', label='real')
                axeslist[traj_index].plot(y[0][traj_index], c='b', label='expected')
                axeslist[traj_index].legend()
    fig.tight_layout()
    fig.savefig(net.save_folder + '{}/'.format(epoch) + 'sanity_check.pdf')
    plt.close(fig)


def error_realtime(net, pars, epoch):
    T = pars['T']
    decays = pars['decays']
    scales = pars['scales']

    bs = 8
    sanity_sampler_params = {
    'n_channels': net.n_channels,
    'epoch_length': T,
    'decays': decays,
    'scales': scales,
    'batch_size': bs,
    'mode': 'train',
    'is_switch': net.is_switch,
    }


    X, y = sample_data(**sanity_sampler_params)
    preds = net.integrate(X)

    logging.critical('in error real time, {} {}'.format(X[0].type, X[0].shape))
    tmp_preds = np.array([t.detach().cpu().numpy() for t in preds])
    tmp_y = np.array([t for t in y])
    logging.critical('in error real time, tmp variables shape {} {}'.format(tmp_preds.shape, tmp_y.shape))
    # , tmp_y = np.stack(x)
    errs = ((tmp_preds-tmp_y)**2).mean(axis=0)
    logging.critical('in error real time, err shape {}'.format(errs.shape))
    os.makedirs(net.save_folder + 'final/error_realtime', exist_ok=True)
    np.savetxt(net.save_folder + 'final/error_realtime/datablob.txt', errs)

    plt.figure()
    for i in range(bs):
        plt.scatter(range(T), errs[i], marker='x')
    plt.yscale('log')
    plt.savefig(net.save_folder + 'final/error_realtime/error_realtime_plot.pdf')
    plt.close()
    # fig.tight_layout()
    # fig.savefig(net.save_folder + '{}/'.format(epoch) + 'sanity_check.pdf')
    # plt.close(fig)

# Spectrum of W, as well as some plots on the weights themselves
def weight_analysis(net, pars, epoch):
    decays = pars['decays']
    scales = pars['scales']

    if net.is_W_parametrized:
        W = net.W.detach()
    elif net.is_2_2_parametrized:
        logging.error('Only W parametrized nets available for now')
        raise RuntimeError

    if net.is_dale_constrained:
        W = net.W.mm(tch.diag(net.synapse_signs)).detach()

    # Total In/Out going weights
    sum_in, sum_out = W.sum(dim=1).cpu().numpy(), W.sum(dim=0).cpu().numpy()
    fig = sns.jointplot(sum_in, sum_out).set_axis_labels("Sum of incoming weights", "Sum of outgoing weights")
    fig.savefig(net.save_folder + '{}/'.format(epoch) + 'in_out_jointplot.pdf')
    plt.close()

    #Plot the weight-matrix, useful mostly for Dale and disjoint initialization
    w_abs = max(W.min().abs(), W.max().abs())
    fig, ax = plt.subplots()
    seismic = plt.get_cmap('seismic')
    sns.heatmap(W.cpu().numpy(), center=0, cmap=seismic, ax=ax)
    # ax.invert_yaxis()
    fig.savefig(net.save_folder + '{}/'.format(epoch) + 'weights_heatmap.png')
    plt.close(fig)


    # Histogram of the weights themselves

    W_np = W.flatten().detach().cpu().numpy()
    W_nonzero = W_np[np.where(W_np!=0)]
    if W_nonzero is None:
        plt.figure()
        plt.axhline(y=0)
        plt.set_xlim(0, 1)
        plt.title('Fraction of zero weights : {}'.format(np.mean(W_np==0)))
        plt.savefig(net.save_folder + '{}/'.format(epoch) + 'weights_histogram.pdf')
        plt.close()
    else:
        plt.figure()
        plt.hist(W_nonzero, bins=30)
        plt.title('Fraction of zero weights : {}'.format(np.mean(W_np==0)))
        plt.savefig(net.save_folder + '{}/'.format(epoch) + 'weights_histogram.pdf')
        plt.close()

        plt.figure()
        plt.hist(W_nonzero, bins=30, log=True)
        plt.title('Fraction of zero weights : {}'.format(np.mean(W_np==0)))
        plt.savefig(net.save_folder + '{}/'.format(epoch) + 'weights_histogram_log.pdf')
        plt.close()


    # Checking if Dale's Law is satisfied
    tmp_1 = np.mean(np.sum(W.cpu().numpy()<=0., axis=0) == net.n)
    tmp_2 = np.mean(np.sum(W.cpu().numpy()>=0., axis=0) == net.n)
    plt.figure()
    plt.scatter(np.mean((W.cpu().numpy() >= 0), axis=0), np.mean((W.cpu().numpy() < 0), axis=0))
    plt.xlabel('Fraction of positive out-going connections')
    plt.ylabel('Fraction of strictly negative out-going connections')
    plt.title('Fraction of only negative out : {}, only positive {}'.format(tmp_1, tmp_2))
    plt.savefig(net.save_folder + '{}/'.format(epoch) + 'dale_satisfaction.pdf')
    plt.close()


    # Study of eigenvalues; only really useful for linear networks, in all other cases svd is more relevant
    # (in particular for ReLU our optimal solution is not diagonalizable)
    if net.saturations == [-1e8, 1e8] and net.activation_type == 'ReLU':
        eigs, _ = tch.eig(W, eigenvectors=False)
        eigs = eigs.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].scatter(eigs[:,0], eigs[:,1], marker='x')
        axes[1].scatter(eigs[:,0], eigs[:,1], marker='x')

        circles = [plt.Circle((0, 0), radius=decays[c], facecolor='None', edgecolor='k', ls='--') for c in range(net.n_channels)]
        for circ in circles:
            axes[0].add_patch(circ)
        circles = [plt.Circle((0, 0), radius=decays[c], facecolor='None', edgecolor='k', ls='--') for c in range(net.n_channels)]
        for circ in circles:
            axes[1].add_patch(circ)

        axes[0].axhline(y=0)
        axes[0].axvline(x=0)
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)

        axes[1].axhline(y=0)
        axes[1].set_xlim(0.92, 1.02)
        axes[1].set_ylim(-0.1, 0.1)

        fig.savefig(net.save_folder + '{}/'.format(epoch) + 'eigs_density.pdf')
        plt.close(fig)

    else:
        U, sigmas, V = tch.svd(W, compute_uv=((epoch=='final') or net.is_dale_constrained))
        sigmas = sigmas.detach().cpu().numpy()
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'sigmas.txt', sigmas)

        plt.figure()
        plt.hist(sigmas, density=False, log=True, bins=100)
        plt.title('First (D+1) s.v. : ' + ', '.join(['{:.3}'.format(s) for s in sigmas[:net.n_channels+1]]))
        plt.savefig(net.save_folder + '{}/'.format(epoch) + 'svd_hist_log.pdf')
        plt.close()

        if epoch == 'final':
            np.save(net.save_folder + 'final/lefts.npy', U[:, :net.n_channels].detach().cpu().numpy())
            np.save(net.save_folder + 'final/rights.npy', V[:, :net.n_channels].detach().cpu().numpy())

        if net.is_dale_constrained:
            exc_idx = range(net.n_excit)
            inh_idx = range(net.n_excit, net.n)
            U = U.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            dots_lr = np.zeros((net.n_channels+1, net.n_channels+1))
            for i in range(net.n_channels+1):
                for j in range(net.n_channels+1):
                    dots_lr[i,j] = V[:, i].dot(U[:, j]).item()
            fig, ax = plt.subplots()
            sns.heatmap(dots_lr, center=0, cmap=seismic, ax=ax, annot=True, fmt=".2")
            fig.savefig(net.save_folder + '{}/'.format(epoch) + 'dot_products_right_left_heatmap.png')
            plt.close(fig)

            dots_dl = np.zeros((net.n_channels, net.n_channels+1))
            for i in range(net.n_channels):
                for j in range(net.n_channels+1):
                    dots_dl[i,j] = net.decoders[i].detach().cpu().numpy().dot(U[:, j]).item()

            fig, ax = plt.subplots()
            sns.heatmap(dots_dl, center=0, cmap=seismic, ax=ax, annot=True, fmt=".2")
            fig.savefig(net.save_folder + '{}/'.format(epoch) + 'dot_products_decoders_left_heatmap.png')
            plt.close(fig)

            for c in range(net.n_channels+1):
                fig, axes = plt.subplots(2)
                axes[0].scatter(exc_idx, U[exc_idx, c], c='r', marker='x', s=4)
                axes[0].scatter(inh_idx, U[inh_idx, c], c='b', marker='x', s=4)
                axes[0].axhline(y=0, ls='--')
                axes[0].set_ylabel('Left eigenvector component')

                axes[1].scatter(exc_idx, V[exc_idx, c], c='r', marker='x', s=4)
                axes[1].scatter(inh_idx, V[inh_idx, c], c='b', marker='x', s=4)
                axes[1].axhline(y=0, ls='--')
                axes[1].set_ylabel('Right eigenvector component')

                fig.savefig(net.save_folder + '{}/'.format(epoch) + 'singular_vectors_structure_{}.pdf'.format(c))

# Look at activity of single neuron
def individual_neuron_activities(net, pars, epoch):
    if epoch != 'final':
        return

    os.makedirs(net.save_folder + 'final/individual_activities/', exist_ok=True)

    T = pars['T']
    decays = pars['decays']
    scales = pars['scales']

    sanity_sampler_params = {
        'n_channels': net.n_channels,
        'epoch_length': T,
        'decays': decays,
        'scales': scales,
        'batch_size': 5,
        'mode': 'test',
        'is_switch': net.is_switch,
    }

    # The test sequences are the same as for sanity_check, so can use the two figures side by side
    # to check that the results do not look at all like the integrals
    X, y = sample_data(**sanity_sampler_params)
    preds, curs = net.integrate(X, keep_currents=True)

    # activation = lambda x: tch.clamp(x, *net.saturations)
    activation = net.activation_function

    for neuron in range(10):
        fig, axeslist = plt.subplots(1, 5, figsize=(5*8, net.n_channels*6))
        for traj_index in range(5):
                axeslist[traj_index].plot(activation(curs[traj_index, :][neuron]).detach().cpu().numpy(), c='b')
        fig.tight_layout()
        fig.savefig(net.save_folder + 'final/individual_activities/neuron_{}.pdf'.format(neuron))
        plt.close(fig)


# Look at the internal representation (in current space)
# Sensitivity plots for individual neurons can only be plotted for D=1 or 2
def fit_internal_representation(net, pars, epoch):
    # if epoch != 'final':
    #     return
    os.makedirs(net.save_folder + 'final/representation_plots', exist_ok=True)
    T = pars['T']
    if net.activation_type == 'Sigmoid':
        T = 400
    decays = pars['decays']
    scales = pars['scales']
    try:
        bs = pars['batch_size']
        if bs > 256:
            bs = (bs//256)*256
    except:
        bs = 128

    big_test_sampler_params = {
    'n_channels': net.n_channels,
    'epoch_length': T,
    'decays': decays,
    'scales': scales,
    'batch_size': min(bs,128),
    'mode': 'test',
    'is_switch': net.is_switch,
    }


    # if bs <= 256:
    X, y = sample_data(**big_test_sampler_params)
    preds, actual_currents = net.integrate(X, keep_currents=True)
    # print(preds[0].shape, actual_currents.shape)
    del X
    # else:
    #     preds = [tch.zeros(bs, T).float() for _ in range(net.n_channels)]
    #     actual_currents = tch.zeros(bs, T, net.n).float()
    #
    #     # X = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
    #     y = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
    #     for i in range(bs//256):
    #         X_tmp, y_tmp = sample_data(**big_test_sampler_params)
    #         preds_tmp, curs_tmp = net.integrate(X_tmp, keep_currents=True)
    #         actual_currents[i*256:(i+1)*256] = curs_tmp
    #
    #         for c in range(net.n_channels):
    #             preds[c][i*256:(i+1)*256] = preds_tmp[c]
    #
    #             # X[c][i*256:(i+1)*256] = X_tmp[c]
    #             y[c][i*256:(i+1)*256] = y_tmp[c]




    W = net.W.detach()
    if net.is_dale_constrained:
        W = net.W.mm(tch.diag(net.synapse_signs)).detach()

    U, _, _ = tch.svd(W, compute_uv=True)
    if not net.is_dale_constrained:
        lefts = U[:, :net.n_channels]
    else:
        lefts = U[:, :net.n_channels+1] #  think one additional singular value is used for balance on top of the "computational" ones
    del U

    actual_currents = actual_currents.reshape((-1, net.n)).transpose(0,1)
    coordinates = utils.lstsq(lefts, actual_currents)[0] #(D, bs *T)
    predicted_currents = lefts.matmul(coordinates)

    # Just print the norm of the current and of the residual in this fit
    logging.critical('Norm of currents in D dim space {}'.format(tch.sqrt(tch.mean(actual_currents**2)).item()))
    logging.critical('Norm of currents orthogonal to D dim space {}'.format(tch.sqrt(tch.mean((actual_currents-predicted_currents)**2)).item()))
    np.savetxt(net.save_folder + 'final/representation_plots/norms_current_fit.txt', np.array([tch.sqrt(tch.mean(actual_currents**2)).item(), tch.sqrt(tch.mean((actual_currents-predicted_currents)**2)).item()]))

    # Check validity of coordinates fit
    for neuron in range(20):
        plt.figure(figsize=(6,6))
        plt.scatter(actual_currents[neuron].detach().cpu().numpy(), predicted_currents[neuron].detach().cpu().numpy(), s=1, rasterized=True)
        plt.plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls='--', c='k', label='$x=y$')
        plt.xlabel('Actual currents')
        plt.ylabel('Predicted currents from fit')
        plt.savefig(net.save_folder + 'final/representation_plots/currents_from_coordinates_fit{}.pdf'.format(neuron))
        plt.close()

    if net.n_channels == 1:
        for neuron in range(10):
            plt.figure(figsize=(6,6))
            plt.scatter(y[0].flatten(), actual_currents[neuron].detach().cpu().numpy(), s=1, rasterized=True)
            # plt.plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls='--', c='k', label='$x=y$')
            plt.xlabel('Actual currents')
            plt.ylabel(r'Value of $y$')
            plt.savefig(net.save_folder + 'final/representation_plots/current_neuron_{}.pdf'.format(neuron))
            plt.close()


    if net.n_channels == 1 and net.is_dale_constrained:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        a, b = coordinates[0].detach().cpu().numpy(), coordinates[1].detach().cpu().numpy()
        axes[0].scatter(y[0].flatten(), a.flatten())
        axes[0].set_xlabel(r'Value of coordinate')
        axes[0].set_ylabel(r'Value of $y$')
        axes[0].set_title('Value of coordinate on top singular value')

        axes[1].scatter(y[0].flatten(), b.flatten())
        axes[1].set_xlabel(r'Value of coordinate')
        axes[1].set_ylabel(r'Value of $y$')
        axes[1].set_title('Value of coordinate on second singular value')

        fig.tight_layout()
        fig.savefig(net.save_folder + 'final/manifold_coordinates_function_of_output.pdf', dpi=600)
        plt.close(fig)
    elif net.n_channels == 2:
        if not net.is_dale_constrained:
            n_coords = 2
            a, b = coordinates[0].detach().cpu().numpy(), coordinates[1].detach().cpu().numpy()
        else:
            n_coords = 3
            a, b, c = coordinates[0].detach().cpu().numpy(), coordinates[1].detach().cpu().numpy(), coordinates[2].detach().cpu().numpy()


        fig, axes = plt.subplots(1, n_coords, figsize=(8, 4))
        seismic = plt.get_cmap('seismic')
        tmp = max(np.abs(a.min()), np.abs(a.max()))
        norm = matplotlib.colors.Normalize(vmin=-tmp, vmax=tmp)
        axes[0].scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(a.flatten())), s=4, rasterized=True)
        axes[0].set_xlabel(r'Value of $y_1$')
        axes[0].set_ylabel(r'Value of $y_2$')
        axes[0].set_title('Value of a')
        divider = make_axes_locatable(axes[0])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)

        tmp = max(np.abs(b.min()), np.abs(b.max()))
        norm = matplotlib.colors.Normalize(vmin=-tmp, vmax=tmp)
        axes[1].scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(b.flatten())), s=4, rasterized=True)
        axes[1].set_xlabel(r'Value of $y_1$')
        axes[1].set_ylabel(r'Value of $y_2$')
        axes[1].set_title('Value of b')
        divider = make_axes_locatable(axes[1])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)

        if net.is_dale_constrained:
            tmp = max(np.abs(b.min()), np.abs(b.max()))
            norm = matplotlib.colors.Normalize(vmin=-tmp, vmax=tmp)
            axes[2].scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(c.flatten())), s=4, rasterized=True)
            axes[2].set_xlabel(r'Value of $y_1$')
            axes[2].set_ylabel(r'Value of $y_2$')
            axes[2].set_title('Value of c')
            divider = make_axes_locatable(axes[1])
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
            fig.add_axes(ax_cb)
        fig.tight_layout()
        fig.savefig(net.save_folder + 'final/manifold_coordinates_function_of_output.pdf', dpi=600)
        plt.close(fig)

    # Fit as a function of output not y, not relevant if the network performs well enough
    try:
        preds = tch.stack(preds, dim=0)
    except:
        pass
    preds = preds.reshape((preds.shape[0], -1)).transpose(0,1)
    representation =  utils.lstsq(preds, coordinates.transpose(0,1))[0]
    predicted_coordinates = preds.matmul(representation).transpose(0,1)

    if net.n_channels == 2 and not net.is_dale_constrained:
        selectivity_vectors = lefts.matmul(representation.transpose(0,1)).detach().cpu().numpy()
        np.save(net.save_folder + 'final/selectivity_vectors.npy', selectivity_vectors)
        angles = np.arctan2(selectivity_vectors[:,1], selectivity_vectors[:,0])
        plt.figure()
        plt.hist(angles, bins=5)
        plt.savefig(net.save_folder + 'final/selectivity_angle_distribution.pdf')
        angles = angles[:10] # Keep for next plot
        del selectivity_vectors

    for c in range(net.n_channels):
        plt.figure(figsize=(6,6))
        plt.scatter(coordinates[c].detach().cpu().numpy(), predicted_coordinates[c].detach().cpu().numpy(), s=1, rasterized=True)
        plt.plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls='--', c='k', label='$x=y$')
        plt.xlabel('Coordinates in the currents representation')
        plt.ylabel('Predicted coordinates from linear fit on the integrals')
        plt.savefig(net.save_folder + 'final/representation_plots/coordinates_from_integrals_fit_channel_{}.pdf'.format(c+1))
        plt.close()


    np.save(net.save_folder + 'final/representation.npy', representation.detach().cpu().numpy())
    del representation, predicted_currents, coordinates

    # activation = lambda x: tch.clamp(x, *net.saturations)
    activation = net.activation_function
    # logging.critical(actual_currents.shape)
    # logging.critical(actual_currents.transpose(1,0).shape)
    real_activities = activation(actual_currents.transpose(1,0)).transpose(1,0).detach().cpu().numpy()
    # real_activities = activation(actual_currents).detach().cpu().numpy()
    del actual_currents
    logging.critical('full_fit shape {}'.format(lefts.matmul(predicted_coordinates).shape))
    full_fit_activities = activation(lefts.matmul(predicted_coordinates).transpose(1,0)).transpose(1,0).detach().cpu().numpy()
    del predicted_coordinates

    for neuron in range(10):
        h = real_activities[neuron]
        plt.figure(figsize=(6,6))
        plt.scatter(h, full_fit_activities[neuron], s=1, rasterized=True)
        plt.plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls='--', c='k', label='$x=y$')
        plt.xlabel("Activity (measured)")
        plt.ylabel("Activity (predicted by two-steps fit)")
        plt.savefig(net.save_folder + 'final/representation_plots/sensitivity_map_{}.pdf'.format(neuron))
        plt.close()


    # This is the part for selectivity plot, can use much larger bs
    if net.n_channels == 2:
        if net.activation_type == 'Sigmoid':
            big_test_sampler_params.update({'mode': 'train', 'epoch_length': 400})
        if bs <= 256:
            X, y = sample_data(**big_test_sampler_params)
            preds, actual_currents = net.integrate(X, keep_currents=True)
            # print(preds[0].shape, actual_currents.shape)
            del X
        else:
            big_test_sampler_params.update({'batch_size': 256})
            preds = [tch.zeros(bs, T).float() for _ in range(net.n_channels)]
            actual_currents = tch.zeros(bs, T, net.n).float()

            # X = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
            y = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
            for i in range(bs//256):
                X_tmp, y_tmp = sample_data(**big_test_sampler_params)
                preds_tmp, curs_tmp = net.integrate(X_tmp, keep_currents=True)
                actual_currents[i*256:(i+1)*256] = curs_tmp

                for c in range(net.n_channels):
                    preds[c][i*256:(i+1)*256] = preds_tmp[c]

                    # X[c][i*256:(i+1)*256] = X_tmp[c]
                    y[c][i*256:(i+1)*256] = y_tmp[c]

        # activation = lambda x: tch.clamp(x, *net.saturations)
        activation = net.activation_function
        actual_currents = actual_currents.reshape((-1, net.n)).transpose(0,1)
        real_activities = activation(actual_currents.to(net.device).transpose(0,1)).transpose(0,1).detach().cpu().numpy()

        del actual_currents

        for neuron in range(10):
            h = real_activities[neuron]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            if not net.activation_type == 'Sigmoid':
                tmp = max(np.abs(h.min()), np.abs(h.max()))
                norm = matplotlib.colors.Normalize(vmin=-tmp, vmax=tmp)
                ax.scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(h.flatten())), s=4, rasterized=True)
            else:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=h.max())
                ax.scatter(y[0].flatten(), y[1].flatten(), c=reds(norm(h.flatten())), s=4, rasterized=True)
            if not net.is_dale_constrained:
                xs = np.linspace(0, plt.gca().get_xlim()[1], 10)
                ys = np.tan(angles[neuron])*xs
                y_lim = plt.gca().get_ylim()
                select = np.logical_and(ys>y_lim[0], ys<y_lim[1])
                ax.plot(xs[select], ys[select])
            # Looks kinda bad because we always plot the x>0 part of the line, which might be in the inactive plane.
            ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y_1$')
            ax.set_ylabel(r'Value of $y_2$')
            ax.set_title('Value of activity')
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            if not net.activation_type == 'Sigmoid':
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
            else:
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=reds, norm=norm, orientation='vertical')

            fig.add_axes(ax_cb)
            fig.savefig(net.save_folder + 'final/representation_plots/selectivity_plot_{}.pdf'.format(neuron))
            plt.close(fig)

    if net.n_channels == 1 and net.activation_type=='Sigmoid':
        # bs = 2048
        big_test_sampler_params.update({'mode': 'test',}) #ensures we reach the highest values of y

        if bs <= 256:
            X, y = sample_data(**big_test_sampler_params)
            preds, actual_currents = net.integrate(X, keep_currents=True)
            # print(preds[0].shape, actual_currents.shape)
            del X
        else:
            big_test_sampler_params.update({'batch_size': 256})
            preds = [tch.zeros(bs, T).float() for _ in range(net.n_channels)]
            actual_currents = tch.zeros(bs, T, net.n).float()

            # X = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
            y = [np.zeros((bs, T)).astype(np.float32) for _ in range(net.n_channels)]
            for i in range(bs//256):
                X_tmp, y_tmp = sample_data(**big_test_sampler_params)
                preds_tmp, curs_tmp = net.integrate(X_tmp, keep_currents=True)
                actual_currents[i*256:(i+1)*256] = curs_tmp

                for c in range(net.n_channels):
                    preds[c][i*256:(i+1)*256] = preds_tmp[c]
                    y[c][i*256:(i+1)*256] = y_tmp[c]

        logging.critical('passed drawing the examples')
        activation = net.activation_function
        actual_currents = actual_currents.reshape((-1, net.n)).transpose(0,1)
        logging.critical(actual_currents.shape)
        real_activities = activation(actual_currents.to(net.device).transpose(1,0)).transpose(1,0).detach().cpu().numpy()


        logging.critical(real_activities.shape)
        for neuron in range(10):
            h = real_activities[neuron]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            ax.scatter(y[0].flatten(), h.flatten(), s=4, rasterized=True)
            # ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y$')
            ax.set_ylabel(r'Value of $h$ (experimental)')
            ax.set_title('Activity map (1D)')
            fig.savefig(net.save_folder + 'final/representation_plots/activity_map_1d_neuron_{}.pdf'.format(neuron))
            plt.close(fig)

        # Also get some saturated ones just to be sure
        mean_act = real_activities.mean(axis=1)
        highest_act_indices = np.argsort(-mean_act)[10:15]
        for neuron in highest_act_indices:
            h = real_activities[neuron]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            ax.scatter(y[0].flatten(), h.flatten(), s=4, rasterized=True)
            # ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y$')
            ax.set_ylabel(r'Value of $h$ (experimental)')
            ax.set_title('Activity map (1D)')
            fig.savefig(net.save_folder + 'final/representation_plots/activity_map_1d_high_act_neuron_{}.pdf'.format(neuron))
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            ax.scatter(y[0].flatten(), actual_currents[neuron].flatten().detach().cpu().numpy(), s=4, rasterized=True)
            # ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y$')
            ax.set_ylabel(r'Value of $h$ (experimental)')
            ax.set_title('Activity map (1D)')
            fig.savefig(net.save_folder + 'final/representation_plots/current_high_act_neuron_{}.pdf'.format(neuron))
            plt.close(fig)


        medium_act_indices = np.argsort(-mean_act)[50:55]
        for neuron in medium_act_indices:
            h = real_activities[neuron]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            ax.scatter(y[0].flatten(), h.flatten(), s=4, rasterized=True)
            # ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y$')
            ax.set_ylabel(r'Value of $h$ (experimental)')
            ax.set_title('Activity map (1D)')
            fig.savefig(net.save_folder + 'final/representation_plots/activity_map_1d_medium_act_neuron_{}.pdf'.format(neuron))
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            ax.scatter(y[0].flatten(), actual_currents[neuron].flatten().detach().cpu().numpy().flatten(), s=4, rasterized=True)
            # ax.set_aspect('equal')
            ax.set_xlabel(r'Value of $y$')
            ax.set_ylabel(r'Value of $h$ (experimental)')
            ax.set_title('Activity map (1D)')
            fig.savefig(net.save_folder + 'final/representation_plots/current_medium_act_neuron_{}.pdf'.format(neuron))
            plt.close(fig)

        del actual_currents

        # logging.critical('finished activity maps')
        n_bins = 100
        y_min, y_max = y[0].min(), y[0].max()
        bins = np.linspace(y_min, y_max, num=n_bins+1)
        agg = np.zeros((n_bins, net.n))
        for b in range(n_bins):
            # logging.critical('before y condition')
            try:
                y[0]>=bins[b]
            except Error as e:
                logging.critical(e)
            y_in_bin = np.logical_and(y[0]>=bins[b], y[0]<bins[b+1])
            # logging.critical('passed y condition')
            # logging.critical('y_in_bin shape {}'.format(y_in_bin.shape))
            neuron_activity_in_bin = real_activities[:, y_in_bin.flatten()]
            # logging.critical('neuron_activity_in_bin shape {}'.format(neuron_activity_in_bin.shape))
            agg[b] = neuron_activity_in_bin.mean(axis=1)

        # logging.critical('passed aggregation')
        mean_act_per_neuron = agg.mean(axis=0)
        reorder = np.argsort(mean_act_per_neuron + 100*(agg[0, :]<=agg[-1, :] ))
        agg = agg[:, reorder]
        # print(agg.shape)
        # logging.critical('passed reordering')

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        seismic = plt.get_cmap('seismic')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # ax.scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(agg.flatten())), s=4, rasterized=True)
        ax.imshow(agg.T, interpolation='nearest', cmap=seismic, norm=norm)

        # Looks kinda bad because we always plot the x>0 part of the line, which might be in the inactive plane.
        ax.set_aspect('auto')
        ax.set_xlabel(r'Value of $y$')
        ax.set_ylabel(r'Index of neuron')
        ax.set_title('Value of activity')
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)
        fig.savefig(net.save_folder + 'final/representation_plots/population_level_coding.pdf')
        plt.close(fig)

        # if epoch =='final':
        np.save(net.save_folder + 'final/agg_for_activity_hists_sigmoid.npy', agg)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        seismic = plt.get_cmap('seismic')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # ax.scatter(y[0].flatten(), y[1].flatten(), c=seismic(norm(agg.flatten())), s=4, rasterized=True)
        for y_bin_idx, y_label, c in zip([50, 75, 99], ['y=0', 'y=2', 'y=4'], ['b', 'g', 'r']):
            acts = agg[y_bin_idx]
            ax.hist(acts, label=y_label, bins=20, log=True, histtype='stepfilled')


        # Looks kinda bad because we always plot the x>0 part of the line, which might be in the inactive plane.
        ax.legend()
        ax.set_aspect('auto')
        ax.set_xlabel(r'Mean neuron activation')
        ax.set_ylabel(r'Index of neuron')
        ax.set_title('Value of activity')
        fig.savefig(net.save_folder + 'final/representation_plots/activity_histograms.pdf')
        plt.close(fig)

def relu_D1_currents(net, pars, epoch):

    assert net.n_channels == 1
    assert net.saturations == [0, 1e8]

    os.makedirs(net.save_folder + 'final/currents_plots', exist_ok=True)
    T = pars['T']
    decays = pars['decays']
    scales = pars['scales']

    big_test_sampler_params = {
    'n_channels': net.n_channels,
    'epoch_length': T,
    'decays': decays,
    'scales': scales,
    'batch_size': 96,
    'mode': 'test',
    'is_switch': net.is_switch,
    }

    X, y = sample_data(**big_test_sampler_params)
    preds, actual_currents = net.integrate(X, keep_currents=True)
    del X
    y = y[0]
    # activation = lambda x: tch.clamp(x, *net.saturations)
    activation = net.activation_function
    states = activation(actual_currents).detach().cpu().numpy()

    l = net.W.matmul(net.encoders[0]).detach()

    # if net.is_dale_constrained:
    #     W = net.W.mm(tch.diag(net.synapse_signs)).detach()
    #     U, sigmas, V = tch.svd(W.detach(), compute_uv=True)
    #     l = U[:, 1]
    #     r = V[:, 1]
    # else:
    #     U, sigmas, V = tch.svd(W.detach(), compute_uv=True)
    #     l = U[:, 0]
    #     r = V[:, 0]
    # del U, V, sigmas

    actual_currents = actual_currents.reshape((-1, net.n)).transpose(0,1)
    # print(l.shape, actual_currents.shape)
    coordinates = utils.lstsq(l.unsqueeze(1), actual_currents)[0] #(D, bs *T)
    # print(coordinates.shape)
    predicted_currents = l.unsqueeze(1).matmul(coordinates)
    # print(predicted_currents, actual_currents)

    plt.figure()
    y_ = y.flatten()
    coordinates = coordinates.flatten()
    # reorder = np.argsort(y)
    # y_, states_= y_[reorder], states_[reorder]
    plt.scatter(actual_currents.detach().cpu().numpy(), predicted_currents.detach().cpu().numpy(), rasterized=True, s=1)
    # a, b = linregress(states)
    plt.savefig(net.save_folder + '{}'.format(epoch) + '/current_fit_validity.pdf')
    plt.close()

    # plt.figure()
    # y_ = y.flatten()
    # coordinates = coordinates.flatten()
    # # reorder = np.argsort(y)
    # # y_, states_= y_[reorder], states_[reorder]
    # sns.kdeplot(actual_currents.detach().cpu().numpy(), predicted_currents.detach().cpu().numpy())
    # # a, b = linregress(states)
    # plt.savefig(net.save_folder + '{}'.format(epoch) + '/current_fit_validity_kde.pdf')
    # plt.close()

    del predicted_currents

    # Coordinate in the current manifold (line here)
    plt.figure()
    y_ = y.flatten()
    coordinates = coordinates.flatten()
    # reorder = np.argsort(y)
    # y_, states_= y_[reorder], states_[reorder]
    # plt.scatter(y_, coordinates.detach().cpu().numpy(), rasterized=True)
    plt.scatter(preds[0].detach().cpu().numpy().flatten(), coordinates.detach().cpu().numpy(), rasterized=True)
    # a, b = linregress(states)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y_, coordinates.detach().cpu().numpy())
    slope, intercept, r_value, p_value, std_err = stats.linregress(preds[0].detach().cpu().numpy().flatten(), coordinates.detach().cpu().numpy())
    plt.plot(y_, slope*y_+intercept, c='k', ls='--')
    plt.savefig(net.save_folder + '{}'.format(epoch) + '/1D_manifold_position.pdf')
    plt.close()
    del y_
    del actual_currents

    if epoch != 'final':
        return

    where_y_pos = np.where(y>0)
    where_y_neg = np.where(y<0)
    y_pos, y_neg = y[where_y_pos], -y[where_y_neg] # negative part of y, add minus sign so it is positive

    states_pos, states_neg = [], []
    for i in range(net.n):
        states_pos.append(states[:,:,i][where_y_pos])
        states_neg.append(states[:,:,i][where_y_neg])

    states_pos = tch.from_numpy(np.array(states_pos)).to(net.device)
    states_neg = tch.from_numpy(np.array(states_neg)).to(net.device)

    pos_uptime = tch.mean((states_pos>0.).float(), dim=1)
    neg_uptime = tch.mean((states_neg>0.).float(), dim=1)
    neuron_colors = ['g' for _ in range(net.n)]
    plt.figure()
    plt.scatter(pos_uptime.detach().cpu().numpy(), neg_uptime.detach().cpu().numpy())
    plt.savefig(net.save_folder + 'final/currents_plots/scatter_uptimes.pdf')
    plt.close('all')

    if not net.is_dale_constrained:
        y_pos_for_fit = tch.from_numpy(y_pos.flatten()).reshape(-1, 1).to(net.device)
        c_pos = lstsq(y_pos_for_fit, states_pos.transpose(0,1))[0].cpu().numpy()
        y_neg_for_fit = tch.from_numpy(y_neg.flatten()).reshape(-1, 1).to(net.device)
        c_neg = lstsq(y_neg_for_fit, states_neg.transpose(0,1))[0].cpu().numpy()
        c_pos, c_neg = c_pos.flatten(), c_neg.flatten()

        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'c_pos.txt', c_pos)
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'c_neg.txt', c_neg)

        pop_idx = (c_pos>1e-3).astype(int) + 2*(c_neg>1e-3).astype(int)
        # Slight reorder, not very important but gives better looking figure
        x_pos = [c_pos[np.where(pop_idx==k)] for k in [1, 2, 3, 0]]
        x_neg = [c_neg[np.where(pop_idx==k)] for k in [1, 2, 3, 0]]

        positives = (pop_idx == 0)
        negatives = (pop_idx == 1)
        shared = (pop_idx == 2)
        nulls = (pop_idx == 3)

        n_pos, n_neg, n_sha, n_nul = np.sum(positives), np.sum(negatives), np.sum(shared), np.sum(nulls)
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'cluster_sizes.txt', [n_pos, n_neg, n_sha, n_nul])
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'encoder.txt', net.encoders[0].detach().cpu().numpy())
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'decoder.txt', net.decoders[0].detach().cpu().numpy())
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'positives.txt', positives)
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'negatives.txt', negatives)
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'shared.txt', shared)
        np.savetxt(net.save_folder + '{}/'.format(epoch) + 'nulls.txt', nulls)

        names = ['Positive', 'Negative', 'Shared', 'Null']
        plt.figure()
        for idx, c in enumerate(['r', 'b', 'g', 'gray']):
            plt.scatter(x_pos[idx], x_neg[idx], color=c, label=names[idx])
        plt.legend()
        plt.savefig(net.save_folder + 'final/currents_plots/scatter_coefs.pdf')
        plt.close('all')

        tmp = ['gray', 'r', 'b', 'g']
        neuron_colors = [tmp[idx] for idx in pop_idx]

        fig, axes = plt.subplots(2, 1, sharex=True)
        xlim = max(c_pos.max(), np.abs(c_neg).max())
        bins = np.linspace(0, xlim, 15)
        axes[0].hist(x_pos, bins, color=['r', 'b', 'g', 'gray'], alpha=0.5, label=['Positive', 'Negative', 'Shared', 'Null'], density=False, log=True)
        axes[1].hist(x_neg, bins, color=['r', 'b', 'g', 'gray'], alpha=0.5, label=['Positive', 'Negative', 'Shared', 'Null'], density=False, log=True)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(net.save_folder + 'final/currents_plots/histogram_coefs.pdf')
        plt.close('all')


    W = net.W.detach()
    if net.is_dale_constrained:
        W = net.W.mm(tch.diag(net.synapse_signs)).detach()

        U, sigmas, V = tch.svd(W.detach(), compute_uv=True)
        l_balance = U[:, 0]
        r_balance = V[:, 0]
        s_balance = sigmas[0]
        l = U[:, 1]
        r = V[:, 1]
        del U, V, sigmas

        exc_idx = range(net.n_excit)
        inh_idx = range(net.n_excit, net.n)

        # "Balance" current
        nu_balance_p = s_balance.detach().cpu().numpy() * l_balance.detach().cpu().numpy() *  r_balance.dot(l*(l>0).float()).detach().cpu().numpy()
        nu_balance_m = s_balance.detach().cpu().numpy() * l_balance.detach().cpu().numpy() *  r_balance.dot(-l*(-l>0).float()).detach().cpu().numpy()

        fig = sns.jointplot(nu_balance_p, nu_balance_m, kind='reg', scatter = False ).set_axis_labels(r"Balance from positive", r"Balance from negative")
        fig.ax_joint.scatter(nu_balance_p, nu_balance_m)
        fig.ax_joint.axvline(x=0, c='k', ls=':')
        fig.ax_joint.axhline(y=0, c='k', ls=':')
        fig.savefig(net.save_folder + 'final/currents_plots/nu_balance.pdf')
        plt.close('all')

    else:
        U, sigmas, V = tch.svd(W.detach(), compute_uv=True)
        l = U[:, 0]
        r = V[:, 0]
        del U, V, sigmas

    nu_e = W.matmul(net.encoders[0]).detach().cpu().numpy() # This is proportional to l, with the small corrections from the bulk
    nu_p = W.matmul(W.matmul(net.encoders[0])*(W.matmul(net.encoders[0])>0).float()).detach().cpu().numpy() / (scales[0] * decays[0])
    nu_m = W.matmul(-W.matmul(net.encoders[0])*(W.matmul(net.encoders[0])<0).float()).detach().cpu().numpy() / (scales[0] * decays[0])

    fig = sns.jointplot(nu_p, nu_m, kind='reg', scatter = False ).set_axis_labels(r"Current from +", r"Current from -")
    fig.ax_joint.scatter(nu_p, nu_m, c=neuron_colors)
    fig.ax_joint.axvline(x=0, c='k', ls=':')
    fig.ax_joint.axhline(y=0, c='k', ls=':')
    fig.savefig(net.save_folder + 'final/currents_plots/nu_p_VS_nu_m.pdf')
    plt.close()

    fig = sns.jointplot(nu_e, nu_p, kind='reg', scatter = False ).set_axis_labels(r"Current from encoder", r"Current from +")
    fig.ax_joint.scatter(nu_e, nu_p, c=neuron_colors)
    fig.ax_joint.axvline(x=0, c='k', ls=':')
    fig.ax_joint.axhline(y=0, c='k', ls=':')
    fig.savefig(net.save_folder + 'final/currents_plots/nu_p_VS_nu_e.pdf')
    plt.close()






tests_register = {
    'weight_analysis': weight_analysis,
    'sanity_check': sanity_check,
    'individual_neuron_activities': individual_neuron_activities,
    'fit_internal_representation': fit_internal_representation,
    'relu_D1_currents': relu_D1_currents,
    'error_realtime': error_realtime,
}
