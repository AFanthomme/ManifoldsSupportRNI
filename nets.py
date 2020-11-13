import torch as tch
import logging
import os
from torch.nn import Module, Parameter, ParameterList, Linear
from torch.nn.functional import sigmoid
from typing import List, Tuple
from torch import Tensor
from numpy import ndarray
import numpy as np

from utils import min, max, orth, sqrtm
from math import sqrt
from scipy.linalg import fractional_matrix_power

# Sample config for ManyChannelsIntegrator:
many_channels_params = {
    'device_name': 'cuda', # Could be changed to run on cpu or specific gpu

    'n': 1000,
    'n_channels': 2,
    'saturations': [0, 1e8],                # This is ReLU; linear is [-1e8, 1e8], Relu4 [0,4]
    'init_radius': 0.,                      # Null initialization
    'save_folder': 'unittests/many_channels_test/',
    # This is for n_channels (= D) > 1
    'init_vectors_type': 'random',          # 'random' is just normed gaussian; 'orthonormal' is with Gram-Schmidt

    # Those ones are only read if D = 1
    'init_vectors_overlap': None,           # Force macroscopic alignment between e and d
    'init_vectors_scales': [1,1],           # Independantly rescale them [d_scale, e_scale]
}


class ManyChannelsIntegrator(Module):
    def __init__(self, args_dict):
        super(ManyChannelsIntegrator, self).__init__()
        self.is_W_parametrized = True
        self.is_dale_constrained = False
        # print('very beginning of init', list(args_dict.items()))
        for k, v in args_dict.items():
             setattr(self, k, v)
        if len(self.saturations) > 2:
            logging.error('ManyChannelsIntegrator.saturations should be [low, high], not {}'.format(saturations))
        std = 1./sqrt(self.n)


        self.n_channels_in = self.n_channels
        self.n_channels_out = self.n_channels
        if self.n_channels == 3 and self.is_switch:
            self.n_channels_out = 1

        self.encoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels_in)])
        self.decoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels_out)])
        self.bias_current = Parameter(tch.zeros(self.n).normal_(0, 0), requires_grad=False)

        if self.init_vectors_type == 'random':
            pass
        elif self.init_vectors_type == 'random_biased':
            self.bias_current.data= .25 *tch.rand(self.n) # bias local field uniform in 0,1
            print('Using random bias current')
        elif self.init_vectors_type == 'orthonormal':
            logging.info('Orthogonalizing encoders and decoders')
            plop = tch.zeros(self.n, self.n_channels_in+self.n_channels_out)
            for idx, item in enumerate(self.encoders):
                plop[:, idx] = item.data
            for idx, item in enumerate(self.decoders):
                plop[:, len(self.encoders)+idx] = item.data
            plop = orth(plop)
            for idx, item in enumerate(self.encoders):
                item.data = plop[:, idx]
            for idx, item in enumerate(self.decoders):
                item.data = plop[:, len(self.encoders)+idx]
        elif self.init_vectors_type == 'support_same':
            self.n_per_channel = self.n // self.n_channels
            delimiters = [c * self.n_per_channel for c in range(self.n_channels)] + [self.n]
            for c in range(self.n_channels):
                self.encoders[c].data = tch.zeros(self.n)
                self.encoders[c].data[delimiters[c]:delimiters[c+1]] = tch.zeros(self.n_per_channel).normal_(0, 1./sqrt(self.n_per_channel))
                self.decoders[c].data = tch.zeros(self.n)
                self.decoders[c].data[delimiters[c]:delimiters[c+1]] = tch.zeros(self.n_per_channel).normal_(0, 1./sqrt(self.n_per_channel))
        elif self.init_vectors_type == 'support_same_with_overlap':
            assert self.n_channels == 2
            self.encoders[0].data = tch.zeros(self.n)
            self.encoders[0].data[:int(.6*self.n)] = tch.zeros(int(.6*self.n)).normal_(0, 1./sqrt(int(.6*self.n)))
            self.decoders[0].data = tch.zeros(self.n)
            self.decoders[0].data[:int(.6*self.n)] = tch.zeros(int(.6*self.n)).normal_(0, 1./sqrt(int(.6*self.n)))
            self.encoders[1].data = tch.zeros(self.n)
            self.encoders[1].data[int(.4*self.n):] = tch.zeros(int(.6*self.n)).normal_(0, 1./sqrt(int(.6*self.n)))
            self.decoders[1].data = tch.zeros(self.n)
            self.decoders[1].data[int(.4*self.n):] = tch.zeros(int(.6*self.n)).normal_(0, 1./sqrt(int(.6*self.n)))
        elif self.init_vectors_type == 'support_disjoint':
            self.n_per_channel = self.n // self.n_channels
            delimiters = [c * self.n_per_channel for c in range(self.n_channels)] + [self.n]
            for c in range(self.n_channels):
                self.decoders[c].data = tch.zeros(self.n)
                self.decoders[c].data[delimiters[c]:delimiters[c+1]] = tch.zeros(self.n_per_channel).normal_(0, 1./sqrt(self.n_per_channel))
                self.encoders[c].data = tch.zeros(self.n)
                lims = (delimiters[c+1],delimiters[c+2]) if c<self.n_channels-1 else (delimiters[0],delimiters[1])
                self.encoders[c].data[lims[0]:lims[1]] = tch.zeros(self.n_per_channel).normal_(0, 1./sqrt(self.n_per_channel))
        elif self.init_vectors_type == 'support_random_e':
            self.n_per_channel = self.n // self.n_channels
            delimiters = [c * self.n_per_channel for c in range(self.n_channels)] + [self.n]
            for c in range(self.n_channels):
                self.decoders[c].data = tch.zeros(self.n)
                self.decoders[c].data[delimiters[c]:delimiters[c+1]] = tch.zeros(self.n_per_channel).normal_(0, 1./sqrt(self.n_per_channel))

        if self.n_channels == 1:
            logging.info('Accessed where enc/dec get normalized')
            # print('In integrator init', self.init_vectors_scales)
            # Force normalizations
            self.encoders[0].data = self.encoders[0].data / tch.sqrt((self.encoders[0].data**2).sum())
            self.decoders[0].data = self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
            # Align the encoder / decoder
            self.decoders[0].data = ((1.-self.init_vectors_overlap)*self.decoders[0].data + self.init_vectors_overlap*self.encoders[0].data)
            # Rescale the io vectors
            self.decoders[0].data = self.init_vectors_scales[0] * self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
            self.encoders[0].data = self.encoders[0].data * self.init_vectors_scales[1]

            logging.info('Measured decoder scale {}'.format(tch.sqrt((self.decoders[0].data**2).sum())))
            logging.info('Desired decoder scale {}'.format(self.init_vectors_scales[0]))
            logging.info(self.init_vectors_scales[1])
        else:
            try:
                for c in range(self.n_channels):
                    self.decoders[c].data = self.init_vectors_scales[0] * self.decoders[c].data
                    self.encoders[c].data = self.init_vectors_scales[1] * self.encoders[c].data
            except AttributeError:
                pass

        self.W = Parameter(tch.zeros(self.n, self.n).normal_(0, std), requires_grad=True)
        eigs, _ = tch.eig(self.W, eigenvectors=False)
        spectral_rad = tch.sqrt((eigs**2).sum(dim=1).max()).item()
        assert spectral_rad != 0
        self.W.data = self.init_radius * self.W.data / spectral_rad
        self.device = tch.device(self.device_name)
        self.to(self.device)

        os.makedirs(self.save_folder, exist_ok=True)
        if self.activation_type == 'ReLU':
            self.activation_function = lambda x: tch.clamp(x, *self.saturations)
        elif self.activation_type == 'Sigmoid':
            if self.sigmoid_random_bias:
                self.thresholds = Parameter(tch.zeros(self.n).uniform_(0, self.sigmoid_threshold).to(self.device), requires_grad=False)
            else:
                self.thresholds = Parameter(tch.ones(self.n).to(self.device) * self.sigmoid_threshold, requires_grad=False)
            if self.sigmoid_train_bias : self.thresholds.requires_grad = True

    def activation_function(self, x):
        if self.activation_type == 'ReLU':
            tmp = tch.clamp(x, *self.saturations)
        elif self.activation_type == 'Sigmoid':
            shape_bkp = x.shape
            # logging.info(x.shape)
            # logging.info(x.shape)
            tmp =  tch.sigmoid(self.sigmoid_slope*(x.view(-1, self.n)-self.thresholds.view(1, self.n)))

        return tmp.view(shape_bkp)


    def step(self, state, inputs, mask, keep_currents=False):
        external_current = self.encoders[0] * inputs[0].view(-1, 1)
        for i in range(1, self.n_channels):
            external_current = external_current + self.encoders[i] * inputs[i].view(-1, 1)
        if keep_currents:
            cur = self.bias_current + (state + mask * external_current).matmul(self.W.t()).detach().clone()


        # state = mask *(self.bias_current + tch.clamp((state + mask * external_current).matmul(self.W.t()), *self.saturations))
        state = mask *(self.bias_current + self.activation_function((state + mask * external_current).matmul(self.W.t())))
        # The .t() above are here for batch operation, but W is really the coupling matrix with correct convention
        # W_ij = weight from j to i
        outs = [(self.decoders[i] * state).sum(-1) for i in range(self.n_channels_out)]
        if keep_currents:
            return outs, state, cur
        else:
            return outs, state

    def forward(self, inputs, state, mask, keep_currents=False):
        T = len(inputs[0][1])
        inputs_unbinded = [inputs[i].unbind(1) for i in range(self.n_channels_in)]
        outputs = [tch.jit.annotate(List[Tensor], []) for _ in range(self.n_channels_out)]
        if keep_currents:
            currents = tch.jit.annotate(List[Tensor], [])
        for t in range(T):
            inp = [inputs_unbinded[i][t] for i in range(self.n_channels_in)]
            if keep_currents:
                outs, state, cur = self.step(state, inp, mask, keep_currents=True)
                currents += [cur.detach()]
            else:
                outs, state = self.step(state, inp, mask, keep_currents=False)
            for i in range(self.n_channels_out):
                outputs[i] = outputs[i] + [outs[i]]
        for i in range(self.n_channels_out):
            outputs[i] = tch.stack(outputs[i], dim=1)
        if keep_currents:
            return outputs, tch.stack(currents, dim=1)
        else:
            return outputs

    def integrate(self, X, keep_currents=False, mask=None):
        # Expect X to be [np.array(bs, T) for c in range(n_channels)]
        if type(X) is not list:
            logging.error('integrate expects a list as X input, not {}'.format(type(X)))
            raise RuntimeError
        if len(X) != self.n_channels_in:
            logging.error('integrate expects same number of input signals as channels, not {} and {}'.format(len(X), self.n_channels_in))
            raise RuntimeError

        # Make the input tch tensor, or do nothing if they already are (e.f. when calling integrate twice on same X)
        for c in range(self.n_channels_in):
            try:
                X[c] = tch.from_numpy(X[c]).to(self.device)
            except TypeError:
                pass

        # mask is not used for this project, but could be useful for implementing "ablations"
        # by forcing a subset of neurons to have 0 activation at all times
        tmp = tch.ones(self.n)
        if mask is not None:
            assert type(mask) is ndarray
            tmp = tch.from_numpy(mask).float()
        mask = tmp.to(self.device)

        init_state = tch.zeros(self.n).to(self.device)
        return self.forward(X, init_state, mask, keep_currents=keep_currents)


class TwoTwoNet(Module):
    # Implementation of net using the 2 by 2 parameters w
    def __init__(self, args_dict):
        super(TwoTwoNet, self).__init__()
        self.is_W_parametrized = False
        self.is_dale_constrained = False
        for k, v in args_dict.items():
             setattr(self, k, v)
        assert self.n_channels == 1
        if len(self.saturations) > 2:
            logging.error('ManyChannelsIntegrator.saturations should be [low, high], not {}'.format(saturations))
        std = 1./sqrt(self.n)
        self.encoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels)])
        self.decoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels)])
        if self.init_vectors_type == 'random':
            pass
        elif self.init_vectors_type == 'orthonormal':
            logging.info('Orthogonalizing encoders and decoders')
            plop = tch.zeros(self.n, 2*self.n_channels)
            for idx, item in enumerate(self.encoders):
                plop[:, idx] = item.data
            for idx, item in enumerate(self.decoders):
                plop[:, len(self.encoders)+idx] = item.data
            plop = orth(plop)
            for idx, item in enumerate(self.encoders):
                item.data = plop[:, idx]
            for idx, item in enumerate(self.decoders):
                item.data = plop[:, len(self.encoders)+idx]

        self.encoders[0].data = self.encoders[0].data / tch.sqrt((self.encoders[0].data**2).sum())
        self.decoders[0].data = self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
        # Align the encoder / decoder
        self.decoders[0].data = ((1.-self.init_vectors_overlap)*self.decoders[0].data + self.init_vectors_overlap*self.encoders[0].data)
        # Rescale the io vectors
        self.decoders[0].data = self.init_vectors_scales[0] * self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
        self.encoders[0].data = self.encoders[0].data * self.init_vectors_scales[1]

        self.w = Parameter(tch.zeros(2, 2).normal_(0, std), requires_grad=True)
        eigs, _ = tch.eig(self.w, eigenvectors=False)
        spectral_rad = tch.sqrt((eigs**2).sum(dim=1).max()).item()
        assert spectral_rad != 0
        self.w.data = self.init_radius * self.w.data / spectral_rad
        self.device = tch.device(self.device_name)
        self.to(self.device)
        os.makedirs(self.save_folder, exist_ok=True)
        self.compute_relevant_quantities()

    def compute_relevant_quantities(self):
        d, e = self.decoders[0].detach().cpu().numpy(), self.encoders[0].detach().cpu().numpy()
        sigma = np.array([[np.dot(d, d), np.dot(d, e)],
                          [np.dot(e, d), np.dot(e, e)]])
        self.sqrt_sigma = fractional_matrix_power(sigma, 1/2)
        self.sqrt_sigma_inv = fractional_matrix_power(sigma, -1/2)

        self.dec_orth = self.sqrt_sigma_inv[0,0] * d + self.sqrt_sigma_inv[0,1] * e
        self.enc_orth  = self.sqrt_sigma_inv[1,0] * d + self.sqrt_sigma_inv[1,1] * e

        assert np.all(np.isfinite(self.sqrt_sigma))
        assert np.all(np.isfinite(self.sqrt_sigma_inv))

        d_d = (np.matmul(self.dec_orth.reshape((-1, 1)), self.dec_orth.reshape((1, -1))))
        d_e = (np.matmul(self.dec_orth.reshape((-1, 1)), self.enc_orth.reshape((1, -1))))
        e_d = (np.matmul(self.enc_orth.reshape((-1, 1)), self.dec_orth.reshape((1, -1))))
        e_e = (np.matmul(self.enc_orth.reshape((-1, 1)), self.enc_orth.reshape((1, -1))))

        # TODO: is this really interesting at all?
        lp, lm = d.dot(d)+e.dot(e), d.dot(d)-e.dot(e)
        e_dot_d = d.dot(e)
        q = np.sqrt(lm**2+4*e_dot_d**2)
        rp, rm = np.sqrt(lp+q), np.sqrt(lp-q)
        b_0 = ((q+lm)*rm+(q-lm)*rp) / (2*e_dot_d*(rp-rm))
        a_0 = -((q-lm)*rm+(q+lm)*rp) / (2*e_dot_d*(rp-rm))

        Z = (e_dot_d*(rp-rm))**2/(2* (q**2))
        self.__dict__.update({'lp': lp, 'lm': lm, 'e_dot_d': e_dot_d, 'q': q, 'rp': rp, 'rm': rm, 'a_0': a_0, 'b_0': b_0, 'Z': Z,
                'sigma': sigma, 'd_d': tch.from_numpy(d_d).to(self.device), 'd_e': tch.from_numpy(d_e).to(self.device),
               'e_d': tch.from_numpy(e_d).to(self.device), 'e_e': tch.from_numpy(e_e).to(self.device)})

    def step(self, state, inputs, mask, keep_currents=False):
        external_current = self.encoders[0] * inputs[0].view(-1, 1)
        for i in range(1, self.n_channels):
            external_current = external_current + self.encoders[i] * inputs[i].view(-1, 1)
        if keep_currents:
            cur = (state + mask * external_current).matmul(self.W.t()).detach().clone()
        state = mask * tch.clamp((state + mask * external_current).matmul(self.W.t()), *self.saturations)
        # The .t() above are here for batch operation, but W is really the coupling matrix with correct convention
        # W_ij = weight from j to i
        outs = [(self.decoders[i] * state).sum(-1) for i in range(self.n_channels)]
        if keep_currents:
            return outs, state, cur
        else:
            return outs, state

    def forward(self, inputs, state, mask, keep_currents=False):
        T = len(inputs[0][1])
        inputs_unbinded = [inputs[i].unbind(1) for i in range(self.n_channels)]
        outputs = [tch.jit.annotate(List[Tensor], []) for _ in range(self.n_channels)]

        # Unpack w to W
        W = self.w[0, 0] * self.d_d
        W = self.w[0, 1] * self.d_e + W
        W = self.w[1, 0] * self.e_d + W
        W = self.w[1, 1] * self.e_e + W
        self.W = W

        if keep_currents:
            currents = tch.jit.annotate(List[Tensor], [])
        for t in range(T):
            inp = [inputs_unbinded[i][t] for i in range(self.n_channels)]
            if keep_currents:
                outs, state, cur = self.step(state, inp, mask, keep_currents=True)
                currents += [cur.detach()]
            else:
                outs, state = self.step(state, inp, mask, keep_currents=False)
            for i in range(self.n_channels):
                outputs[i] = outputs[i] + [outs[i]]
        for i in range(self.n_channels):
            outputs[i] = tch.stack(outputs[i], dim=1)

        # Remove W
        self.W = None

        if keep_currents:
            return outputs, tch.stack(currents, dim=1)
        else:
            return outputs

    def integrate(self, X, keep_currents=False, mask=None):
        # Expect X to be [np.array(bs, T) for c in range(n_channels)]
        if type(X) is not list:
            logging.error('integrate expects a list as X input, not {}'.format(type(X)))
            raise RuntimeError
        if len(X) != self.n_channels:
            logging.error('integrate expects same number of input signals as channels, not {} and {}'.format(len(X), self.n_channels))
            raise RuntimeError

        # Make the input tch tensor, or do nothing if they already are (e.f. when calling integrate twice on same X)
        for c in range(self.n_channels):
            try:
                X[c] = tch.from_numpy(X[c]).to(self.device)
            except TypeError:
                pass

        # mask is not used for this project, but could be useful for implementing "ablations"
        # by forcing a subset of neurons to have 0 activation at all times
        tmp = tch.ones(self.n)
        if mask is not None:
            assert type(mask) is ndarray
            tmp = tch.from_numpy(mask).float()
        mask = tmp.to(self.device)

        init_state = tch.zeros(self.n).to(self.device)
        return self.forward(X, init_state, mask, keep_currents=keep_currents)


class DaleConstrainedIntegrator(Module):
    def __init__(self, args_dict):
        super(DaleConstrainedIntegrator, self).__init__()
        self.is_W_parametrized = True
        self.is_dale_constrained = True

        for k, v in args_dict.items():
             setattr(self, k, v)
        if self.saturations != [0, 1e8]:
            logging.error('DaleConstrainedIntegrators should be ReLU, not saturated as {}'.format(self.saturations))
            raise RuntimeError

        std = 1./sqrt(self.n)

        # Dale specific parameters
        # self.inhib_proportion = .25 # Fraction of neurons that will be inhibitory, should now be a parameter
        # Don't add that yet...
        # self.inhib_fan_out = 20 # Number of allowed out-going connections for inhibitory neurons
        # self.excit_fan_out = 20 # Number of allowed out-going connections for excitatory neurons




        self.encoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels)])
        self.decoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels)])
        if self.init_vectors_type == 'random':
            pass
        elif self.init_vectors_type == 'orthonormal':
            logging.info('Orthogonalizing encoders and decoders')
            plop = tch.zeros(self.n, 2*self.n_channels)
            for idx, item in enumerate(self.encoders):
                plop[:, idx] = item.data
            for idx, item in enumerate(self.decoders):
                plop[:, len(self.encoders)+idx] = item.data
            plop = orth(plop)
            for idx, item in enumerate(self.encoders):
                item.data = plop[:, idx]
            for idx, item in enumerate(self.decoders):
                item.data = plop[:, len(self.encoders)+idx]

        if self.n_channels == 1:
            # Force normalizations
            self.encoders[0].data = self.encoders[0].data / tch.sqrt((self.encoders[0].data**2).sum())
            self.decoders[0].data = self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
            # Align the encoder / decoder
            self.decoders[0].data = ((1.-self.init_vectors_overlap)*self.decoders[0].data + self.init_vectors_overlap*self.encoders[0].data)
            # Rescale the io vectors
            self.decoders[0].data = self.init_vectors_scales[0] * self.decoders[0].data / tch.sqrt((self.decoders[0].data**2).sum())
            self.encoders[0].data = self.encoders[0].data * self.init_vectors_scales[1]



        self.n_inhib = int(self.n * self.inhib_proportion)
        self.n_excit = self.n - self.n_inhib
        self.synapse_signs = Parameter(tch.Tensor([1. for _ in range(self.n_excit)] + [-1. for _ in range(self.n_inhib)]), requires_grad=False).float()
        self.W = Parameter(tch.zeros(self.n, self.n).normal_(0, std), requires_grad=True)
        eigs, _ = tch.eig(self.W, eigenvectors=False)
        spectral_rad = tch.sqrt((eigs**2).sum(dim=1).max()).item()
        assert spectral_rad != 0
        self.W.data = self.init_radius * self.W.data / spectral_rad
        if self.init_radius != 0:
            logging.error('DaleConstrainedIntegrators should be initialized with W=0 for now at least')
            raise RuntimeError
        assert (self.W.data == 0.).all()
        self.device = tch.device(self.device_name)
        self.to(self.device)
        os.makedirs(self.save_folder, exist_ok=True)

    def step(self, state, inputs, mask, keep_currents=False):
        external_current = self.encoders[0] * inputs[0].view(-1, 1)
        for i in range(1, self.n_channels):
            external_current = external_current + self.encoders[i] * inputs[i].view(-1, 1)
        if keep_currents:
            cur = (state + mask * external_current).matmul((self.W.matmul(tch.diag(self.synapse_signs))).t()).detach().clone()
        state = mask * tch.clamp((state + mask * external_current).matmul((self.W.matmul(tch.diag(self.synapse_signs))).t()), *self.saturations)
        # The .t() above are here for batch operation, but W is really the coupling matrix with correct convention
        # W_ij = weight from j to i
        outs = [(self.decoders[i] * state).sum(-1) for i in range(self.n_channels)]
        if keep_currents:
            return outs, state, cur
        else:
            return outs, state

    def forward(self, inputs, state, mask, keep_currents=False):
        T = len(inputs[0][1])
        inputs_unbinded = [inputs[i].unbind(1) for i in range(self.n_channels)]
        outputs = [tch.jit.annotate(List[Tensor], []) for _ in range(self.n_channels)]
        if keep_currents:
            currents = tch.jit.annotate(List[Tensor], [])
        for t in range(T):
            inp = [inputs_unbinded[i][t] for i in range(self.n_channels)]
            if keep_currents:
                outs, state, cur = self.step(state, inp, mask, keep_currents=True)
                currents += [cur.detach()]
            else:
                outs, state = self.step(state, inp, mask, keep_currents=False)
            for i in range(self.n_channels):
                outputs[i] = outputs[i] + [outs[i]]
        for i in range(self.n_channels):
            outputs[i] = tch.stack(outputs[i], dim=1)
        if keep_currents:
            return outputs, tch.stack(currents, dim=1)
        else:
            return outputs

    def integrate(self, X, keep_currents=False, mask=None):
        # Expect X to be [np.array(bs, T) for c in range(n_channels)]
        if type(X) is not list:
            logging.error('integrate expects a list as X input, not {}'.format(type(X)))
            raise RuntimeError
        if len(X) != self.n_channels:
            logging.error('integrate expects same number of input signals as channels, not {} and {}'.format(len(X), self.n_channels))
            raise RuntimeError
        if not (self.W >= 0.).all():
            logging.error('Found non fully positive W in integrate, something went wrong in optimization')
            raise RuntimeError


        # Make the input tch tensor, or do nothing if they already are (e.f. when calling integrate twice on same X)
        for c in range(self.n_channels):
            try:
                X[c] = tch.from_numpy(X[c]).to(self.device)
            except TypeError:
                pass

        # mask is not used for this project, but could be useful for implementing "ablations"
        # by forcing a subset of neurons to have 0 activation at all times
        tmp = tch.ones(self.n)
        if mask is not None:
            assert type(mask) is ndarray
            tmp = tch.from_numpy(mask).float()
        mask = tmp.to(self.device)

        init_state = tch.zeros(self.n).to(self.device)
        return self.forward(X, init_state, mask, keep_currents=keep_currents)


class ManyChannelsIntegratorNonLinearDecoder(Module):
    def __init__(self, args_dict):
        super(ManyChannelsIntegratorNonLinearDecoder, self).__init__()
        self.is_non_linear_decoder = True
        self.is_W_parametrized = True
        self.is_dale_constrained = False
        # print('very beginning of init', list(args_dict.items()))
        for k, v in args_dict.items():
             setattr(self, k, v)
        if len(self.saturations) > 2:
            logging.error('ManyChannelsIntegrator.saturations should be [low, high], not {}'.format(saturations))
        std = 1./sqrt(self.n)


        self.n_channels_in = self.n_channels
        self.n_channels_out = self.n_channels

        self.encoders = ParameterList([Parameter(tch.zeros(self.n).normal_(0, std), requires_grad=False) for _ in range(self.n_channels_in)])
        self.decoder1 = tch.nn.Linear(self.n, self.n_channels_out)
        # self.decoder1 = tch.nn.Linear(self.n, 512)
        # self.decoder2 = tch.nn.Linear(512, self.n_channels_out)
        self.bias_current = Parameter(tch.zeros(self.n).normal_(0, 0), requires_grad=False)


        self.W = Parameter(tch.zeros(self.n, self.n).normal_(0, std), requires_grad=True)
        eigs, _ = tch.eig(self.W, eigenvectors=False)
        spectral_rad = tch.sqrt((eigs**2).sum(dim=1).max()).item()
        assert spectral_rad != 0
        self.W.data = self.init_radius * self.W.data / spectral_rad
        self.device = tch.device(self.device_name)
        self.to(self.device)

        os.makedirs(self.save_folder, exist_ok=True)
        if self.activation_type == 'ReLU':
            self.activation_function = lambda x: tch.clamp(x, *self.saturations)
        elif self.activation_type == 'Sigmoid':
            if self.sigmoid_random_bias:
                self.thresholds = Parameter(tch.zeros(self.n).uniform_(0, self.sigmoid_threshold).to(self.device), requires_grad=False)
            else:
                self.thresholds = Parameter(tch.ones(self.n).to(self.device) * self.sigmoid_threshold, requires_grad=False)
            if self.sigmoid_train_bias : self.thresholds.requires_grad = True

    def activation_function(self, x):
        shape_bkp = x.shape
        if self.activation_type == 'ReLU':
            tmp = tch.clamp(x, *self.saturations)
        elif self.activation_type == 'Sigmoid':
            # logging.info(x.shape)
            # no variable threshold, its a pain otherwise to reuse it for decoding layer
            tmp =  tch.sigmoid(self.sigmoid_slope*(x-self.sigmoid_threshold))

        return tmp

    def decode(self, state):
        return self.activation_function(self.decoder1(state))
        # return self.decoder2(self.activation_function(self.decoder1(state)))

    def step(self, state, inputs, mask, keep_currents=False):
        external_current = self.encoders[0] * inputs[0].view(-1, 1)
        for i in range(1, self.n_channels):
            external_current = external_current + self.encoders[i] * inputs[i].view(-1, 1)
        if keep_currents:
            cur = self.bias_current + (state + mask * external_current).matmul(self.W.t()).detach().clone()


        # state = mask *(self.bias_current + tch.clamp((state + mask * external_current).matmul(self.W.t()), *self.saturations))
        state = mask *(self.bias_current + self.activation_function((state + mask * external_current).matmul(self.W.t())))
        # The .t() above are here for batch operation, but W is really the coupling matrix with correct convention
        # W_ij = weight from j to i
        # outs = [(self.decoders[i] * state).sum(-1) for i in range(self.n_channels_out)]
        outs = self.decode(state)
        # logging.info(outs.shape)
        if keep_currents:
            return outs, state, cur
        else:
            return outs, state

    def forward(self, inputs, state, mask, keep_currents=False):
        T = len(inputs[0][1])
        inputs_unbinded = [inputs[i].unbind(1) for i in range(self.n_channels_in)]
        # outputs = [tch.jit.annotate(List[Tensor], []) for _ in range(self.n_channels_out)]
        if keep_currents:
            currents = tch.jit.annotate(List[Tensor], [])
        for t in range(T):
            inp = [inputs_unbinded[i][t] for i in range(self.n_channels_in)]
            if keep_currents:
                outs, state, cur = self.step(state, inp, mask, keep_currents=True)
                currents += [cur.detach()]
            else:
                outs, state = self.step(state, inp, mask, keep_currents=False)
            # for i in range(self.n_channels_out):
            #     outputs[i] = outputs[i] + [outs[i]]
            if t == 0:
                outputs = outs.unsqueeze(-1)
            else:
                outputs = tch.cat((outputs, outs.unsqueeze(-1)), -1)
        outputs = outputs.permute(1, 0, 2)
        logging.info(outputs.shape)
        # for i in range(self.n_channels_out):
        #     outputs[i] = tch.stack(outputs[i], dim=1)
        if keep_currents:
            return outputs, tch.stack(currents, dim=1)
        else:
            return outputs

    def integrate(self, X, keep_currents=False, mask=None):
        # Expect X to be [np.array(bs, T) for c in range(n_channels)]
        if type(X) is not list:
            logging.error('integrate expects a list as X input, not {}'.format(type(X)))
            raise RuntimeError
        if len(X) != self.n_channels_in:
            logging.error('integrate expects same number of input signals as channels, not {} and {}'.format(len(X), self.n_channels_in))
            raise RuntimeError

        # Make the input tch tensor, or do nothing if they already are (e.f. when calling integrate twice on same X)
        for c in range(self.n_channels_in):
            try:
                X[c] = tch.from_numpy(X[c]).to(self.device)
            except TypeError:
                pass

        # mask is not used for this project, but could be useful for implementing "ablations"
        # by forcing a subset of neurons to have 0 activation at all times
        tmp = tch.ones(self.n)
        if mask is not None:
            assert type(mask) is ndarray
            tmp = tch.from_numpy(mask).float()
        mask = tmp.to(self.device)

        init_state = tch.zeros(self.n).to(self.device)
        return self.forward(X, init_state, mask, keep_currents=keep_currents)
