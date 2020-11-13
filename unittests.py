# Tests on the different components of the program.
import os
import logging
import numpy as np
import torch as tch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from datagen import simple_decay, sample_data


logging.basicConfig(level=logging.INFO)

# Check our implementation of scipy.linalg.orth in torch
def test_orth_torch():
    import utils
    from scipy.linalg import orth

    dists_sci_cpu = []
    dists_sci_gpu = []
    for _ in range(5):
        A = np.random.randn(100, 100)
        orth_A_sci = orth(A)
        orth_A_tch = utils.orth(tch.from_numpy(A)).numpy()
        orth_A_tch_gpu = utils.orth(tch.from_numpy(A).to(tch.device('cuda'))).cpu().numpy()
        dists_sci_cpu.append(np.abs(orth_A_sci-orth_A_tch).max())
        dists_sci_gpu.append(np.abs(orth_A_sci-orth_A_tch_gpu).max())

    assert np.max(dists_sci_cpu) < 2*tch.finfo(tch.float32).eps
    assert np.max(dists_sci_gpu) < 2*tch.finfo(tch.float32).eps

    dists_sci_cpu = ['{:.2e}'.format(d) for d in dists_sci_cpu]
    dists_sci_gpu = ['{:.2e}'.format(d) for d in dists_sci_gpu]
    logging.info('Distances between scipy and torch cpu implementation of orth : {}'.format(dists_sci_cpu))
    logging.info('Distances between scipy and torch gpu implementation of orth: {}'.format(dists_sci_gpu))


    try:
        utils.orth(tch.from_numpy(np.random.randn(4, 5, 6))).numpy()
    except RuntimeError:
        logging.info('orth failed as expected for non 2d inputs')


# Check our implementation of scipy.linalg.sqrtm in torch
def test_sqrtm_torch():
    import utils
    from scipy.linalg import sqrtm

    dists_sci_cpu = []
    dists_sci_gpu = []
    for _ in range(5):
        A = np.random.randn(100, 100)
        # Make A symmetric and Positive-Definite
        A = A + A.transpose()
        plop = np.linalg.eigh(A)[0].min()
        if plop < 0:
            A = A + np.diag([-plop+1e-8 for _ in range(100)])
        sqrtm_A_sci = sqrtm(A)
        sqrtm_A_tch = utils.sqrtm(tch.from_numpy(A)).numpy()
        sqrtm_A_tch_gpu = utils.sqrtm(tch.from_numpy(A).to(tch.device('cuda'))).cpu().numpy()
        dists_sci_cpu.append(np.abs(sqrtm_A_sci-sqrtm_A_tch).max())
        dists_sci_gpu.append(np.abs(sqrtm_A_sci-sqrtm_A_tch_gpu).max())

    assert np.max(dists_sci_cpu) < 2*tch.finfo(tch.float32).eps
    assert np.max(dists_sci_gpu) < 2*tch.finfo(tch.float32).eps

    dists_sci_cpu = ['{:.2e}'.format(d) for d in dists_sci_cpu]
    dists_sci_gpu = ['{:.2e}'.format(d) for d in dists_sci_gpu]
    logging.info('Distances between scipy and torch cpu implementation of sqrtm: {}'.format(dists_sci_cpu))
    logging.info('Distances between scipy and torch gpu implementation of sqrtm: {}'.format(dists_sci_gpu))


    try:
        utils.sqrtm(tch.from_numpy(np.random.randn(4, 5, 6))).numpy()
    except RuntimeError:
        logging.info('sqrtm failed as expected for non 2d inputs')

    try:
        utils.sqrtm(tch.from_numpy(np.random.randn(4, 5,))).numpy()
    except RuntimeError:
        logging.info('sqrtm failed as expected for non square inputs')

    try:
        utils.sqrtm(tch.from_numpy(np.random.randn(4, 4,))).numpy()
    except RuntimeError:
        logging.info('sqrtm failed as expected for non symmetric inputs')

    try:
        A = np.random.randn(4, 4,)
        A = A + A.transpose()
        utils.sqrtm(tch.from_numpy(A)).numpy()
    except RuntimeError:
        logging.info('sqrtm failed as expected for non PD inputs')

# Print the first test trajectories
def plot_test_sequences():
    from datagen import simple_decay
    trajs = np.load('precomputed/test_sequences/batch_0.npy')
    targets = simple_decay(trajs, 0.995)
    os.makedirs('unittests/datagen/precomputed', exist_ok=True)
    for traj_idx in range(5):
        fig, ax = plt.subplots()
        ax.plot(trajs[traj_idx], c='b')
        ax.set_ylabel('Inputs')
        twin_ax = ax.twinx()
        twin_ax.plot(targets[traj_idx], c='r')
        twin_ax.set_ylabel('Targets')
        ax.set_xlabel('Time')
        fig.savefig('unittests/datagen/precomputed/traj_{}.pdf'.format(traj_idx))

    # Check that indeed simple_decay fails when called on non-batched sequences
    try:
        simple_decay(trajs[0], .995)
    except RuntimeError:
        logging.info('simple_decay failed as expected for non-batched inputs')

# Check both train and test mode of data sampler
def test_data_sampler():
    from datagen import sample_data
    os.makedirs('unittests/datagen/sampler', exist_ok=True)
    decays = [.97, .99, .99]
    scales = [1., 1., .2]
    n_channels = 3

    X, y = sample_data(n_channels=n_channels, decays=decays, scales=scales, epoch_length=500, mode='train')
    for traj_idx in range(5):
        fig, axes = plt.subplots(n_channels)
        for c in range(n_channels):
            ax = axes[c]
            ax.scatter(range(len(X[c][traj_idx])), X[c][traj_idx], c='b')
            ax.set_ylabel('Inputs (channel {})'.format(c+1))
            twin_ax = ax.twinx()
            twin_ax.plot(y[c][traj_idx], c='r')
            twin_ax.set_ylabel('Targets (channel {})'.format(c+1))
            ax.set_xlabel('Time')
        fig.savefig('unittests/datagen/sampler/train_traj_{}.pdf'.format(traj_idx))

    X, y = sample_data(n_channels=n_channels, decays=decays, scales=scales, epoch_length=500, mode='test')
    for traj_idx in range(5):
        fig, axes = plt.subplots(n_channels)
        for c in range(n_channels):
            ax = axes[c]
            ax.scatter(range(len(X[c][traj_idx])), X[c][traj_idx], c='b')
            ax.set_ylabel('Inputs (channel {})'.format(c+1))
            twin_ax = ax.twinx()
            twin_ax.plot(y[c][traj_idx], c='r')
            twin_ax.set_ylabel('Targets (channel {})'.format(c+1))
            ax.set_xlabel('Time')
        fig.savefig('unittests/datagen/sampler/test_traj_{}.pdf'.format(traj_idx))

    try:
        sample_data(n_channels=n_channels, decays=decays, scales=scales, epoch_length=1500, mode='test')
    except RuntimeError:
        logging.info('simple_decay failed as expected for too long test trajectories')

    try:
        sample_data(n_channels=n_channels, decays=decays, scales=scales, epoch_length=200, batch_size=2000, mode='test')
    except RuntimeError:
        logging.info('simple_decay failed as expected for too large test batches')

    try:
        sample_data(n_channels=n_channels, decays=decays+[0.5], scales=scales, epoch_length=200, batch_size=2000, mode='test')
    except RuntimeError:
        logging.info('simple_decay failed as expected for mismatch in number of decays')

    try:
        sample_data(n_channels=n_channels, decays=decays, scales=scales[:-1], epoch_length=200, batch_size=2000, mode='test')
    except RuntimeError:
        logging.info('simple_decay failed as expected for mismatch in number of scales')

# Check the interfaces of ManyChannelsIntegrator
def test_many_channels_net():
    from datagen import sample_data
    from nets import ManyChannelsIntegrator, many_channels_params
    import torch
    net = ManyChannelsIntegrator(many_channels_params)
    X, _ = sample_data(n_channels=2, decays=[.97, .99], scales=[1., 1.], epoch_length=500, mode='train')
    out = net.integrate(X)
    logging.info('net.integrate worked')
    out_cur, cur = net.integrate(X, keep_currents=True)
    logging.info('net.integrate with keep_currents=True worked')
    out_masked, _ = net.integrate(X, keep_currents=True, mask=np.ones(net.n))
    logging.info('net.integrate with keep_currents=True and dummy mask worked')
    for c in range(2):
        assert (out[c]==out_cur[c]).all()
        assert (out[c]==out_masked[c]).all()
    logging.info('net.integrate gave the same result with all optional keyword arguments')

    bad_X = np.array(X)
    try:
        net.integrate(bad_X)
    except RuntimeError:
        logging.info('integrate failed as expected for non list-formatted inputs')

    bad_X, _ = sample_data(n_channels=3, decays=[.95, .97, .99], scales=[1.,1.,1.], epoch_length=500, mode='train')
    try:
        net.integrate(bad_X)
    except RuntimeError:
        logging.info('integrate failed as expected for number of channels mismatch')

    pars = deepcopy(many_channels_params)
    pars['init_vectors_type'] = 'orthonormal'
    net = ManyChannelsIntegrator(pars)
    logging.info('net.init worked with init_vector_type = orthonormal')
    vects = [*net.encoders, *net.decoders]
    dots = tch.Tensor([[vects[i].dot(vects[j]) for i in range(net.n_channels)] for j in range(net.n_channels)])

    logging.info('Maximum difference between dots matrix and identity : {}'.format((dots-tch.eye(net.n_channels)).abs().max()))

# Check batch_loss works well
def test_batch_loss():
    from losses import batch_loss
    from nets import ManyChannelsIntegrator, many_channels_params
    net = ManyChannelsIntegrator(many_channels_params)
    D = net.n_channels
    decays = [.995 for c in range(D)]
    scales = [10. for c in range(D)]
    batch_size = 256
    epoch_length = 20

    sampler_params = {
    'n_channels': D,
    'epoch_length': 20,
    'decays': [.995 for _ in range(D)],
    'scales': [1. for _ in range(D)],
    'batch_size': 256,
    }

    loss = batch_loss(net, **sampler_params)
    logging.info('batch_loss_many_channels worked')

def test_lstsq():
    import numpy as np
    import utils

    for _ in range(10):
        A = np.random.randn(10, 3) # np M = mon n = 10; np N = mon D = 3
        b = np.random.randn(10, 1000) # np K = mon bs*T = 1000
        bob = np.linalg.lstsq(A,b)[0]
        A = tch.from_numpy(A)
        b = tch.from_numpy(b)
        pop = utils.lstsq(A,b)[0].numpy()
        logging.info('Max difference between utils.lstsq and numpy.lstsq : {}'.format(np.max(np.abs(bob-pop))))
