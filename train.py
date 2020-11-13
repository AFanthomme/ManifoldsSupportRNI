import torch as tch
from torch.optim import Adam, SGD
from copy import deepcopy
import numpy as np
from math import sqrt
import os
import logging
import warnings

# Only here for the loss in realtime plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from losses import batch_loss, average_loss_D1, switch_loss, average_loss_D2, average_loss_generic, average_loss_generic_non_linear_decoder
from nets import ManyChannelsIntegrator, many_channels_params, TwoTwoNet, DaleConstrainedIntegrator, ManyChannelsIntegratorNonLinearDecoder
from datagen import sample_data, sampler_params
from tests import tests_register

train_params = {
    'loss_name': 'batch',
    'optimizer_name': 'adam',
    'lr': 5e-4,
    'n_epochs': 10000,
    'stop_loss': 1e-8,
}

test_suite = {
    'weight_analysis': {'period': 1000},
}

full_params = {
    'net_type': 'many_channels',
    'net_params': many_channels_params,
    'sampler_params': sampler_params,
    'train_params': train_params,
    'test_suite': test_suite,
    'rescale_s_dot': False,
    'rescale_s_norms': False,
}



def main(full_params, seed):
    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    np.random.seed(seed)


    net_params = full_params['net_params']
    net_params['save_folder'] = net_params['save_folder'] + 'seed{}/'.format(seed)

    sampler_params = full_params['sampler_params']
    train_params = full_params['train_params']
    test_suite = full_params['test_suite']

    # be careful, this works but relies on mutability of dicts, should instead call init(net_params) !!
    if full_params['net_type'] == 'many_channels':
        # print('Just before calling init', list(net_params.items()))
        # print('Just before calling init', full_params['net_params']['init_vector_scales'])
        # net = ManyChannelsIntegrator(full_params['net_params'])
        net = ManyChannelsIntegrator(net_params)
    elif full_params['net_type'] == 'TwoTwoNet':
        net = TwoTwoNet(full_params['net_params'])
    elif full_params['net_type'] == 'DaleNet':
        logging.info('Using Dale constrained network')
        net = DaleConstrainedIntegrator(full_params['net_params'])
    elif full_params['net_type'] == 'NonLinearDecoder':
        logging.info('Using NonLinearDecoder network')
        net = ManyChannelsIntegratorNonLinearDecoder(full_params['net_params'])
    else:
        logging.error('Only many_channels is allowed as net_type for now')
        raise RuntimeError

    if not full_params['net_type'] == 'NonLinearDecoder':
        if train_params['train_ed']:
            # print('encoders max', net.encoders[0].abs().max().item())
            # print('decoders max', net.decoders[0].abs().max().item())
            # print('w max', net.W.abs().max().item())

            for c in range(net.n_channels):
                net.decoders[c].requires_grad =  True
                net.encoders[c].requires_grad =  True
            decoders_start_normed = [d/np.sqrt(d.dot(d).item()) for d in net.decoders]
            encoders_start_normed = [e/np.sqrt(e.dot(e).item()) for e in net.encoders]
            if train_params['train_d_only']:
                raise RuntimeError('Choose one between train e+d and train d only')

        elif train_params['train_d_only']:
            for c in range(net.n_channels):
                net.decoders[c].requires_grad =  True
            decoders_start_normed = [d/np.sqrt(d.dot(d).item()) for d in net.decoders]
            encoders_normed = [e/np.sqrt(e.dot(e).item()) for e in net.encoders]
            # This is garbage, just use d_scale
            # if train_params['d_init'] == 'small':
            #     for c in range(net.n_channels):
            #         net.decoders[c] *= 1e-2

        # These are useful mostly for single channel networks, but they could be used someday
        # Need to be done after drawing encoders/decoders so cannot be put in params directly.
        if net.n_channels == 1:
            if full_params['rescale_s_dot']:
                s_before = deepcopy(sampler_params['scales'])
                d_dot_e = [net.encoders[c].dot(net.decoders[c]).detach().item() for c in range(net.n_channels)]
                sampler_params['scales'] = [s_before[c] * d_dot_e[c] for c in range(net.n_channels)]
                logging.critical('Renormalized scales by dot product [seed{}]'.format(seed))
            elif full_params['rescale_s_norms']:
                s_before = deepcopy(sampler_params['scales'])
                norms_enc = [net.encoders[c].dot(net.encoders[c]).detach().item() for c in range(net.n_channels)]
                norms_dec = [net.decoders[c].dot(net.encoders[c]).detach().item() for c in range(net.n_channels)]
                norm_prod = [sqrt(norms_enc[c] * norms_dec[c]) for c in range(net.n_channels)]
                sampler_params['scales'] = [s_before[c] * norm_prod[c] for c in range(net.n_channels)]
                logging.critical('Renormalized scales by product of norms [seed{}]'.format(seed))

    for _, test_params in test_suite.items():
        test_params['scales'] = sampler_params['scales']
        test_params['decays'] = sampler_params['decays']


    if train_params['loss_name'] == 'batch':
        loss_function = batch_loss
    elif train_params['loss_name'] == 'switch_loss':
        loss_function = switch_loss
    elif train_params['loss_name'] == 'avg_d1':
        if net.n_channels != 1:
            logging.error('Average loss only allowed for single channel case')
            raise RuntimeError
        else:
            loss_function = average_loss_D1
    elif train_params['loss_name'] == 'avg_d2':
        if net.n_channels != 2:
            logging.error('Average loss D2 only allowed for bi-channel case')
            raise RuntimeError
        else:
            loss_function = average_loss_D2
    elif train_params['loss_name'] == 'avg_generic':
        loss_function = average_loss_generic
    elif train_params['loss_name'] == 'avg_generic_non_linear_decoder':
        loss_function = average_loss_generic_non_linear_decoder
    else:
        logging.error("Invalid loss name {}".format(train_params['loss_name']))
        raise RuntimeError

    if train_params['optimizer_name'] == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=train_params['lr'])
    elif train_params['optimizer_name'] == 'adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=train_params['lr'])
    else:
        logging.error('Only sgd and adam optimizers are implemented, not {}'.format(train_params['optimizer_name']))
        raise RuntimeError

    # logging.error('hello')
    # for name, param in net.named_parameters():
    #     logging.error(name)
    #     logging.error(param.requires_grad)
    #
    # for param in net.parameters():
    #     logging.error('data shape {}, requires_grad {}'.format(param.data.shape, param.requires_grad))

    logging.info('The optimizer is working on the following (1000,1000) is W, 1000 is d or e :')
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k is 'params':
                outputs += (k + ': ')
                for vp in v:
                    outputs += (str(vp.shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        logging.info(outputs)
    # logging.error(optimizer.param_groups)

    # Not very elegant but it does the trick
    def do_tests(net, epoch):
        if (epoch == 0) or (test_suite is None):
            return
        for test_name, test_params in test_suite.items():
            try:
                if epoch == 'final':
                    raise TypeError
                should_launch = (epoch % test_params['period'] == 0)
            except TypeError:
                should_launch = True

            if should_launch:
                print('Launching test {}'.format(test_name))
                try:
                    os.makedirs(net.save_folder + '{}/'.format(epoch), exist_ok=True)
                except FileExistsError:
                    pass
                tests_register[test_name](net, test_params, epoch)
                print('Done running test {}'.format(test_name))

    # Keep the best candidate on cpu to limit gpu memory usage
    saved_net = deepcopy(net).cpu()
    saved_loss = 1e8
    best_loss = 1e8
    n_epochs = train_params['n_epochs']
    stop_loss = train_params['stop_loss']
    losses = np.zeros(n_epochs)

    if not full_params['net_type'] == 'NonLinearDecoder':
        if train_params['train_d_only']:
            # if this is a use_switch network, it will fail, but that's fine
            d_scales = np.zeros((net.n_channels, n_epochs))
            d_d0_normalized = np.zeros((net.n_channels, n_epochs))
            d_e_normalized = np.zeros((net.n_channels, n_epochs))
        if train_params['train_ed']:
            # if this is a use_switch network, it will fail, but that's fine
            d_scales = np.zeros((net.n_channels, n_epochs))
            d_d0_normalized = np.zeros((net.n_channels, n_epochs))
            d_e_normalized = np.zeros((net.n_channels, n_epochs))
            e_scales = np.zeros((net.n_channels, n_epochs))
            e_e0_normalized = np.zeros((net.n_channels, n_epochs))


    for epoch in range(n_epochs):
        loss = loss_function(net, **sampler_params) # Loss functions only use parameters related to data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if full_params['net_type'] == 'DaleNet':
            net.W.data.clamp_(0)

        if full_params['net_type'] == 'DaleNet':
            # print('assertion test for daleNet')
            assert (net.W.data>=0.).all()

        losses[epoch] = loss.detach().item()

        if not full_params['net_type'] == 'NonLinearDecoder':

            if train_params['train_d_only']:
                for c in range(net.n_channels):
                    d_scales[c, epoch] = np.sqrt(net.decoders[c].dot(net.decoders[c]).item())
                    d_d0_normalized[c, epoch] = net.decoders[c].dot(decoders_start_normed[c]).item() / (d_scales[c,epoch]+1e-8)
                    d_e_normalized[c, epoch] = net.decoders[c].dot(encoders_normed[c]).item() / (d_scales[c,epoch]+1e-8)
            elif train_params['train_ed']:
                for c in range(net.n_channels):
                    d_scales[c, epoch] = np.sqrt(net.decoders[c].dot(net.decoders[c]).item())
                    e_scales[c, epoch] = np.sqrt(net.encoders[c].dot(net.encoders[c]).item())
                    d_d0_normalized[c, epoch] = net.decoders[c].dot(decoders_start_normed[c]).item() / (d_scales[c,epoch]+1e-8)
                    e_e0_normalized[c, epoch] = net.encoders[c].dot(encoders_start_normed[c]).item() / (d_scales[c,epoch]+1e-8)
                    d_e_normalized[c, epoch] = net.decoders[c].dot(net.encoders[c]).item() / (d_scales[c,epoch]*e_scales[c,epoch]+1e-8)


        if losses[epoch] > 1e2:
            logging.critical('Loss became much too large, stopped training')
            with open(net.save_folder + 'FAILED.txt', mode='w+') as f:
                f.write('Failed at training epoch {}'.format(epoch))
            return

        # store best_net on cpu to free some gpu memory
        # .75 here to avoid having to transfer model back and forth to cpu at every step...
        if losses[epoch] < .75 * saved_loss:
            saved_net = deepcopy(net).cpu()
            saved_loss = losses[epoch]

        # Always test on current net state
        with tch.set_grad_enabled(False):
            do_tests(net, epoch)

        if losses[epoch] < best_loss:
            best_loss = losses[epoch]
            logging.info('New best loss {} at step {}, T{}[seed{}]'.format(losses[epoch], epoch,
                                        sampler_params['epoch_length'], seed))

        if epoch % 100 == 0:
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore') # Suppress the warning for missing glyph
            plt.figure()
            plt.semilogy(losses[:epoch])
            plt.savefig(net.save_folder + 'loss.png')
            plt.close()
            # with open(net.save_folder + 'losses.txt', "ab") as f:
            np.savetxt(net.save_folder + 'losses.txt', losses)

            if not full_params['net_type'] == 'NonLinearDecoder':
                if train_params['train_d_only'] or train_params['train_ed']:
                    for c in range(net.n_channels):
                        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                        ax[0].plot(d_scales[c, :epoch])
                        ax[0].set_xlabel('Training step')
                        ax[0].set_ylabel('Norm of d')
                        ax[1].plot(d_d0_normalized[c, :epoch])
                        ax[1].set_xlabel('Training step')
                        ax[1].set_ylabel('Normalized dot product d.d_start')
                        ax[2].plot(d_e_normalized[c, :epoch])
                        ax[2].set_xlabel('Training step')
                        ax[2].set_ylabel('Normalized dot product d.e')
                        fig.savefig(net.save_folder + 'd_evolution_channel_{}.png'.format(c))
                        plt.close(fig)
                if train_params['train_ed']:
                    for c in range(net.n_channels):
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].plot(e_scales[c, :epoch])
                        ax[0].set_xlabel('Training step')
                        ax[0].set_ylabel('Norm of e')
                        ax[1].plot(e_e0_normalized[c, :epoch])
                        ax[1].set_xlabel('Training step')
                        ax[1].set_ylabel('Normalized dot product e.e_start')
                        fig.savefig(net.save_folder + 'e_evolution_channel_{}.png'.format(c))
                        plt.close(fig)

            # with open(net.save_folder + 'losses.txt', "ab") as f:
            np.savetxt(net.save_folder + 'losses.txt', losses)
        if losses[epoch] <= stop_loss:
            losses[epoch:] = losses[epoch]
            break


    if losses[epoch] > saved_loss:
        net = saved_net.to(net.device)

    # This is not the best way to save model, but it works fine for our needs
    bkp = net.activation_function
    net.activation_function = None
    tch.save(net, net.save_folder + 'best_net.pt')
    net.activation_function = bkp
    do_tests(net, 'final')
