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
from nets import ManyChannelsIntegrator, many_channels_params, TwoTwoNet, DaleConstrainedIntegrator
from datagen import sample_data, sampler_params
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

base_folder = 'out/D3_sigmoid_batch/n_1000_slope_50.0_thresh_0.1_train_bias_True/'



os.makedirs('out/mante/', exist_ok=True)
class SigmoidHead(tch.nn.Module):
    def __init__(self):
        super(SigmoidHead, self).__init__()
        self.head_vector = tch.nn.Parameter(tch.zeros(n).normal_(0, 1./np.sqrt(n)).cuda(), requires_grad=True)

    def forward(self, x):
        # tmp = x.matmul(self.head_vector)
        tmp = x.matmul(self.head_vector)
        # print(tmp.shape)
        return tch.sigmoid(50*(tmp-.1))

for seed in range(4):
    bs = 1024
    epoch_length = 40
    T = epoch_length
    n = 1000
    sampler_pars = deepcopy(sampler_params)
    sampler_pars.update({'n_channels': 3, 'decays': [.8, .75, .75], 'scales': [1., 1., 1.], 'batch_size': bs, 'epoch_length': epoch_length, 'mode': 'train'})
    f = base_folder + 'seed{}/'.format(seed)
    net = tch.load(f+'best_net.pt')
    net.activation_function = lambda x: tch.sigmoid(net.sigmoid_slope*(x.view(-1, n)-net.thresholds.view(1, net.n))).view(x.shape)
    head = SigmoidHead()
    # print(head.head_vector)
    opt = Adam([head.head_vector], lr=5e-3)

    for epoch in range(1000):
        with tch.no_grad():
            X, y = sample_data(**sampler_pars)

            # This is for the final classification thing
            # probs = .5 * tch.ones(bs)
            # X[2] = tch.bernoulli(probs).unsqueeze(-1).repeat(1, epoch_length).to(net.device)
            # print(X[2].shape)

            _, curs = net.integrate(X, keep_currents=True)
            # This is for the final classification thing
            # states = net.activation_function(curs)[:, -1, :]
            # target_out = np.zeros(bs)
            # channels = X[2][:, 0]
            states = net.activation_function(curs)

            channels = y[2] > 0.
            target_out = np.zeros((bs, T))
            for b in range(bs):
                # idx = int(channels[b].item())
                # target_out[b] = (y[idx][b, -1] > 0)
                for t in range(epoch_length):
                    idx = int(channels[b, t].item())
                    target_out[b, t] = (y[idx][b, t] > 0)

        # print(states.shape)
        out = head(states)
        loss = tch.nn.MSELoss()(out, tch.from_numpy(target_out).float().to(net.device))
        out_binary = (out >.5).detach().cpu().numpy()
        if epoch % 10 == 0:
            print('Step {}: loss {:.3e}, error rate {}'.format(epoch, loss, 1. - np.mean(out_binary == target_out)))
        opt.zero_grad()
        loss.backward()
        opt.step()

    os.makedirs('out/mante/seed{}'.format(seed), exist_ok=True)
    test_T = 40
    sampler_pars.update({'batch_size': 20, 'epoch_length': test_T, 'mode': 'train'})
    X, y = sample_data(**sampler_pars)
    preds, curs = net.integrate(X, keep_currents=True)
    states = net.activation_function(curs)
    out = head(states)
    channels = y[2] > 0.
    target_out = np.zeros((20, test_T))
    indices = np.zeros((20, test_T))
    for b in range(20):
        # idx = int(channels[b].item())
        # target_out[b] = (y[idx][b, -1] > 0)
        for t in range(test_T):
            idx = int(channels[b, t].item())
            indices[b,t] = idx
            target_out[b, t] = (y[idx][b, t] > 0)


    for i in range(20):
        fig, axes = plt.subplots(2, sharex=True)
        ax = axes[0]
        # ax.plot(preds[0][i].detach().cpu().numpy(), c='r', label='$y_0$')
        # ax.plot(preds[1][i].detach().cpu().numpy(), c='b', label='$y_1$')
        ax.scatter(range(test_T), preds[0][i].detach().cpu().numpy(), c='r', marker='x', label='$y_0$')
        ax.scatter(range(test_T), preds[1][i].detach().cpu().numpy(), c='b', marker='x', label='$y_1$')
        bkg_idx = int(indices[i, 0])
        idx_start = 0
        draw = False
        collections = [[], []]
        try:
            print(indices[i])


            for t in range(idx_start, test_T+1):
                if indices[i, t] != bkg_idx:
                    new_bkg_idx = int(indices[i, t])
                    new_idx_start = int(t)
                    draw=True
                if draw:
                    print('draw at idx {}'.format(t))
                    draw=False
                    # print(ax.get_ylim())
                    # print(ax.get_ylim()[1] - ax.get_ylim()[0])
                    collections[bkg_idx].append(Rectangle((idx_start, ax.get_ylim()[0]), new_idx_start-idx_start, ax.get_ylim()[1] - ax.get_ylim()[0]))
                    idx_start = new_idx_start
                    bkg_idx = new_bkg_idx
        except IndexError:
            # if (collections[0] is None and collections[1] is None):
            collections[bkg_idx].append(Rectangle((idx_start, ax.get_ylim()[0]), test_T-idx_start, ax.get_ylim()[1] - ax.get_ylim()[0]))

        print(collections[0], collections[1])
        if collections[0] is not None:
            pc0 = PatchCollection(collections[0], facecolor='r', alpha=.5, edgecolor=None)
            ax.add_collection(pc0)
        if collections[1] is not None:
            pc1 = PatchCollection(collections[1], facecolor='b', alpha=.5, edgecolor=None)
            ax.add_collection(pc1)
        ax.axhline(y=0, c='k', ls='--')
        ax.legend()

        ax = axes[1]
        print(out.shape)
        ax.scatter(range(test_T), out[i].detach().cpu().numpy(), c='k')
        # ax.plot(out[i].detach().cpu().numpy(), c='k')
        ax.axhline(y=.5, ls='--', c='k')
        if collections[0] is not None:
            pc0 = PatchCollection(collections[0], facecolor='r', alpha=.5, edgecolor=None)
            ax.add_collection(pc0)
        if collections[1] is not None:
            pc1 = PatchCollection(collections[1], facecolor='b', alpha=.5, edgecolor=None)
            ax.add_collection(pc1)

        f = 'out/mante/seed{}/traj{}.pdf'.format(seed, i)
        fig.savefig(f)
