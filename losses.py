import torch as tch
import logging

from datagen import sample_data

# Batch loss, same for all nets
def batch_loss(net, **sampler_params):
    X, y = sample_data(mode='train', **sampler_params)
    scales = sampler_params['scales']
    T = sampler_params['epoch_length']
    preds = net.integrate(X)
    loss = 0
    for i in range(net.n_channels):
        loss = loss + tch.nn.MSELoss()(preds[i], tch.from_numpy(y[i]).to(net.device)) / (scales[i]**2 * (T**2))

    if net.is_dale_constrained:
        l2 = (net.W**2).mean()
        loss = loss + net.l2_penalty * l2

    return loss

def switch_loss(net, **sampler_params):
    # works only with 2 channels
    assert sampler_params['n_channels'] == 3
    assert sampler_params['is_switch'] == True
    X, y = sample_data(mode='train', **sampler_params)

    scales = sampler_params['scales']
    T = sampler_params['epoch_length']
    preds = net.integrate(X)
    loss = 0
    for i in range(net.n_channels_out):
        loss = loss + tch.nn.MSELoss()(preds[i], tch.from_numpy(y[i]).to(net.device)) / (scales[i]**2 * (T**2))

    if net.is_dale_constrained:
        l2 = (net.W**2).mean()
        loss = loss + net.l2_penalty * l2

    return loss

# Average loss for D=1, either the true one (linear) or the proxy
def average_loss_D1(net, **sampler_params):
    assert net.n_channels == 1
    decay = sampler_params['decays'][0]
    s = sampler_params['scales'][0]

    if net.is_W_parametrized:
        W = net.W
    elif net.is_2_2_parametrized:
        W = net.w[0, 0] * net.d_d
        W = net.w[0, 1] * net.d_e + W
        W = net.w[1, 0] * net.e_d + W
        W = net.w[1, 1] * net.e_e + W


    if net.is_dale_constrained:
        assert (net.W>=0).all()
        W = net.W.matmul(tch.diag(net.synapse_signs))

    if net.saturations == [-1e8, 1e8] and net.activation_type == 'ReLU':
        k, T = 1, sampler_params['epoch_length']
        acc = 0
        W_k_e = W.matmul(net.encoders[0])
        while k <= T:
            acc = acc + (T+1-k) * (net.decoders[0].dot(W_k_e) - s * (decay**k))**2
            W_k_e = W.matmul(W_k_e)
            k += 1
        return  acc / ((T**2)*(s**2))

    elif net.saturations == [0, 1e8] and net.activation_type == 'ReLU':
        p_W_k_e = tch.clamp(W.matmul(net.encoders[0]), 0, 1e8)
        m_W_k_e = tch.clamp(-W.matmul(net.encoders[0]), 0, 1e8)

        loss_sync = tch.sum((W.matmul(p_W_k_e) - decay * W.matmul(net.encoders[0]))**2)
        loss_sync = loss_sync + tch.sum((W.matmul(m_W_k_e) + decay * W.matmul(net.encoders[0]))**2)

        p_acc = (net.decoders[0].dot(p_W_k_e) - s * decay)**2
        m_acc = (net.decoders[0].dot(m_W_k_e) + s * decay)**2

        return (p_acc+m_acc+loss_sync) / (s**2)
    else:
        activation = net.activation_function
        range = (-5., 5.) # This is for our decay=.8 test, I think it will be enough
        W_e = W.matmul(net.encoders[0]).view(1, -1)
        alphas = tch.zeros(4096,1).uniform_(*range).to(net.device)
        # logging.critical('{}  {}  {}'.format((W_e**2).mean(), (alphas**2).mean(), (alphas.matmul(W_e)**2).mean()))

        curs = alphas.matmul(W_e)
        dots = activation(curs).matmul(net.decoders[0])
        dots_expected = s*decay*alphas

        new_states = activation(curs).matmul(W.t())
        new_states_expected = decay*curs
        # logging.critical('{}  {}  {}  {}'.format(dots.shape,dots_expected.shape, new_states.shape, new_states_expected.shape))

        loss = 0
        loss = loss + ((dots - dots_expected.squeeze(1))**2).mean()
        loss = loss + ((new_states - new_states_expected)**2).mean()
        return loss


def average_loss_D2(net, **sampler_params):
    assert net.n_channels == 2
    decay1, decay2 = sampler_params['decays']
    s1, s2 = sampler_params['scales']

    if net.is_W_parametrized:
        W = net.W
    elif net.is_2_2_parametrized:
        W = net.w[0, 0] * net.d_d
        W = net.w[0, 1] * net.d_e + W
        W = net.w[1, 0] * net.e_d + W
        W = net.w[1, 1] * net.e_e + W

    if net.is_dale_constrained:
        assert (net.W>=0).all()
        W = net.W.matmul(tch.diag(net.synapse_signs))

    # Same loss no matter what
    activation = net.activation_function
    range = (-5., 5.) # This is for our decay=.8 test, I think it will be enough
    W_e1 = W.matmul(net.encoders[0]).view(1, -1)
    W_e2 = W.matmul(net.encoders[1]).view(1, -1)
    alphas = tch.zeros(4096,2).uniform_(*range).to(net.device)
    # logging.critical('{}  {}  {}'.format((W_e**2).mean(), (alphas**2).mean(), (alphas.matmul(W_e)**2).mean()))

    curs = alphas[:,0].unsqueeze(1).matmul(W_e1) + alphas[:,1].unsqueeze(1).matmul(W_e2)
    dots1 = activation(curs).matmul(net.decoders[0])
    dots2 = activation(curs).matmul(net.decoders[1])
    dots1_expected = s1*decay1*alphas[:,0]
    dots2_expected = s2*decay2*alphas[:,1]

    new_curs = activation(curs).matmul(W.t())
    new_curs_expected = decay1 * alphas[:,0].unsqueeze(1).matmul(W_e1) + decay2 * alphas[:,1].unsqueeze(1).matmul(W_e2)
    # logging.critical('{}  {}  {}  {}'.format(dots.shape,dots_expected.shape, new_states.shape, new_states_expected.shape))

    loss = 0
    loss = loss + ((dots1 - dots1_expected)**2).mean()
    loss = loss + ((dots2 - dots2_expected)**2).mean()
    loss = loss + ((new_curs - new_curs_expected)**2).mean()

    return loss

def average_loss_generic(net, **sampler_params):
    d= net.n_channels
    decays = sampler_params['decays']
    scales = sampler_params['scales']

    if net.is_W_parametrized:
        W = net.W
    elif net.is_2_2_parametrized:
        W = net.w[0, 0] * net.d_d
        W = net.w[0, 1] * net.d_e + W
        W = net.w[1, 0] * net.e_d + W
        W = net.w[1, 1] * net.e_e + W

    if net.is_dale_constrained:
        assert (net.W>=0).all()
        W = net.W.matmul(tch.diag(net.synapse_signs))

    # Same loss no matter what
    activation = net.activation_function
    z_range = (-5.1, 5.1)
    Wes = [W.matmul(e).view(1, -1) for e in net.encoders]

    alphas = tch.zeros(sampler_params['batch_size'],d).uniform_(*z_range).to(net.device)
    # logging.critical('{}  {}  {}'.format((W_e**2).mean(), (alphas**2).mean(), (alphas.matmul(W_e)**2).mean()))


    curs = alphas[:,0].unsqueeze(1).matmul(Wes[0])
    for i in range(1, d):
        curs = curs + alphas[:,i].unsqueeze(1).matmul(Wes[i])

    dots = [activation(curs).matmul(d) for d in net.decoders]
    # dots2 = activation(curs).matmul(net.decoders[1])
    dots_expected = [scales[i]*decays[i]*alphas[:,i] for i in range(d)]
    # dots2_expected = s2*decay2*alphas[:,1]

    new_curs = activation(curs).matmul(W.t())
    new_curs_expected = decays[0] * alphas[:,0].unsqueeze(1).matmul(Wes[0])
    for i in range(1, d):
        new_curs_expected = new_curs_expected + decays[i] * alphas[:,i].unsqueeze(1).matmul(Wes[i])
    # logging.critical('{}  {}  {}  {}'.format(dots.shape,dots_expected.shape, new_states.shape, new_states_expected.shape))

    loss = 0
    for i in range(d):
        loss = loss + ((dots[i] - dots_expected[i])**2).mean()

    # logging.critical('Dots loss {}, curs loss {}'.format(loss.item(),  ((new_curs - new_curs_expected)**2).mean()))
    loss = loss + ((new_curs - new_curs_expected)**2).mean()

    return loss


def average_loss_generic_non_linear_decoder(net, **sampler_params):
    d= net.n_channels
    assert net.is_non_linear_decoder
    decays = sampler_params['decays']
    scales = sampler_params['scales']

    if net.is_W_parametrized:
        W = net.W
    elif net.is_2_2_parametrized:
        W = net.w[0, 0] * net.d_d
        W = net.w[0, 1] * net.d_e + W
        W = net.w[1, 0] * net.e_d + W
        W = net.w[1, 1] * net.e_e + W

    if net.is_dale_constrained:
        assert (net.W>=0).all()
        W = net.W.matmul(tch.diag(net.synapse_signs))

    # Same loss no matter what
    activation = net.activation_function
    z_range = (-5.1, 5.1)
    Wes = [W.matmul(e).view(1, -1) for e in net.encoders]

    alphas = tch.zeros(sampler_params['batch_size'],d).uniform_(*z_range).to(net.device)
    # logging.critical('{}  {}  {}'.format((W_e**2).mean(), (alphas**2).mean(), (alphas.matmul(W_e)**2).mean()))


    curs = alphas[:,0].unsqueeze(1).matmul(Wes[0])
    for i in range(1, d):
        curs = curs + alphas[:,i].unsqueeze(1).matmul(Wes[i])

    dots = net.decode(activation(curs)).transpose(0,1)
    # logging.critical(dots.shape)
    # dots2 = activation(curs).matmul(net.decoders[1])
    dots_expected = [scales[i]*decays[i]*alphas[:,i] for i in range(d)]
    # dots2_expected = s2*decay2*alphas[:,1]

    new_curs = activation(curs).matmul(W.t())
    new_curs_expected = decays[0] * alphas[:,0].unsqueeze(1).matmul(Wes[0])
    for i in range(1, d):
        new_curs_expected = new_curs_expected + decays[i] * alphas[:,i].unsqueeze(1).matmul(Wes[i])
    # logging.critical('{}  {}  {}  {}'.format(dots.shape,dots_expected.shape, new_states.shape, new_states_expected.shape))

    loss = 0
    for i in range(d):
        loss = loss + tch.nn.MSELoss(dots[i], dots_expected[i])

    # logging.critical('Dots loss {}, curs loss {}'.format(loss.item(),  ((new_curs - new_curs_expected)**2).mean()))
    loss = loss + tch.nn.MSELoss(new_curs, new_curs_expected)

    return loss
