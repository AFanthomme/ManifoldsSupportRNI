# Pure numpy here
import numpy as np
from copy import deepcopy
import os
import logging
from utils import min

class TestSequencesPrecomputer:
    # This should be called from setup.py, could be adapted if of interest
    # In that case, will just_change the content of test_sequences and do as if nothing happened
    def __init__(self):
        self.base_rate = .05
        self.burst_length = 25
        self.burst_rate = .95
        self.refractory_length = 10
        self.n_bursts = 20
        self.epoch_length = 1000

        self.rng = np.random.RandomState(0)

    def generate_one_burst(self, rate, len):
        return self.rng.choice([0,1], size=(len,), p=[1-rate, rate])

    def _full_profile(self):
        # Return ONE ternary sequence with elements {+1, -1, 0}
        start_times = np.sort(self.rng.choice(np.arange(self.epoch_length-self.burst_length-1), self.n_bursts))
        ref = start_times[0]
        filtered_times = [ref]
        current_pos = 0

        starts, ends = [], []
        profile = np.zeros((self.epoch_length, 2))

        for start_time in start_times:
            if start_time - ref < self.burst_length + self.refractory_length:
                pass
            else:
                filtered_times.append(start_time)
                ref = start_time

        for idx, t in enumerate(filtered_times):
            direction = self.rng.choice([0, 1])
            profile[t:t+self.burst_length, direction] += self.generate_one_burst(self.burst_rate, self.burst_length)

        # Out-of-burst periods:
        ref = 0
        for idx, t in enumerate(filtered_times):
            profile[ref:t] = self.rng.choice([0, 1], size=[t-ref, 2], p=[1.-self.base_rate, self.base_rate])
            ref = t + self.burst_length
        profile[ref:self.epoch_length] = np.random.choice([0, 1], size=[self.epoch_length-ref, 2], p=[1.-self.base_rate, self.base_rate])
        return profile[:,0]-profile[:,1]

    def _build_trajs(self, bs=256, n_trajs=100):
        logging.info('Generating test trajectories, takes around 1 minute')
        for traj_idx in range(n_trajs):
            out = np.zeros((bs, self.epoch_length), dtype=np.float32)
            for idx in range(bs):
                out[idx] = self._full_profile()
            np.save('precomputed/test_sequences/batch_{}.npy'.format(traj_idx), out)


def simple_decay(data, decay):
    if len(data.shape) == 1:
        logging.error('datagen.simple_decay need batches of shape (bs, T), not {}'.format(data.shape))
        raise RuntimeError
    out = np.zeros_like(data)
    for t in range(data.shape[1]):
        out[:, t] = decay * (data[:, t] +  out[:, t-1])
    return out


sampler_params = {
'n_channels': 1,
'epoch_length': 1000,
'decays': [.995],
'scales': [1.],
'batch_size': 256,
'mode': 'train',
'is_switch' : False,
}

def sample_data(n_channels=1, epoch_length=1000, decays=[.995], scales=[1.], batch_size=256, mode='train', is_switch=False):
    if type(batch_size) is not int:
        logging.error('sample_data expects integer input')
        raise RuntimeError
    if len(decays) != n_channels:
        logging.error('sample_data expects as many decays as channels, not {} and {}'.format(len(decays), n_channels))
        raise RuntimeError
    if len(scales) != n_channels:
        logging.error('sample_data expects as many scales as channels, not {} and {}'.format(len(scales), n_channels))
        raise RuntimeError

    if mode == 'train':
        if is_switch:
            X = [np.random.randn(batch_size, epoch_length).astype(np.float32) for _ in range(2)]
            X += [np.random.randint(2, size=(batch_size,1)).repeat(epoch_length, axis=1).astype(np.float32)]
        else:
            X = [np.random.randn(batch_size, epoch_length).astype(np.float32) for _ in range(n_channels)]
    elif mode == 'test':
        # This part to be modified if longer sequences are interesting
        if epoch_length > 1000:
            logging.error('sample_data expects T<1000 in test mode, not {}'.format(epoch_length))
            raise RuntimeError
        if batch_size > 256:
            logging.error('sample_data expects batch_size<256 in test mode, not {}'.format(batch_size))
            raise RuntimeError

        X = [np.load('precomputed/test_sequences/batch_{}.npy'.format(c)).astype(np.float32)[:batch_size, :epoch_length] for c in range(n_channels)]
        for x_ in X:
            np.random.shuffle(x_)


    if is_switch:
        X_mixed = np.where(X[-1]==0., X[0], X[1]) # condition, table if cond=True, table if cond=False
        # logging.error(X_mixed.shape)
        y = [scales[c] * simple_decay(X_mixed, decays[c]) for c in range(1)]
    else:
        y = [scales[c] * simple_decay(X[c], decays[c]) for c in range(n_channels)]
    logging.debug('Data sampling went as expected')
    return X, y
