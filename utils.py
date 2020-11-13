# Pytorch implementation of some useful linalg functions from scipy
import torch as tch
import logging
from scipy.special import logsumexp
stable_add = lambda log_abs, signs: logsumexp(a=log_abs, b=signs, return_sign=True)

max = lambda a,b: a if a>b else b
min = lambda a,b: a if a<b else b

# pytorch implementation of vector family orthonormalization
def orth(A, rcond=None):
    if len(A.shape) != 2:
        logging.error('Expected input to orth be a matrix, not {}d tensor'.format(len(A.shape)))
        raise RuntimeError
    u, s, v = tch.svd(A, some=True)
    v.transpose_(0,1)
    M, N = u.shape[0], v.shape[1]
    if rcond is None:
        rcond = tch.finfo(s.dtype).eps * max(M, N)
    tol = s.max() * rcond
    num = tch.sum(s > tol)
    Q = u[:, :num]
    return Q

# pytorch implementation of matrix square root
def sqrtm(A):
    if len(A.shape) != 2:
        logging.error('Expected input to sqrtm be a matrix, not {}d tensor'.format(len(A.shape)))
        raise RuntimeError

    if A.shape[0] != A.shape[1]:
        logging.error('Expected input to sqrtm be square, not {}'.format(A.shape))
        raise RuntimeError

    if not (A.transpose(0, 1) == A).all():
        logging.error('Expected input to sqrtm to be symmetric')
        raise RuntimeError

    e, V = tch.symeig(A, eigenvectors=True)

    if not (e>=0).all():
        logging.error('Calling sqrtm on a matrix with negative eigenvalues, min eig {}'.format(e.min()))
        raise RuntimeError

    return V.matmul((tch.diag(e)**.5).matmul(V.t()))

# lstsq has been added in latest pytorch (1.4.0), but due to drivers incompatibility we use an earlier one
def lstsq(A,b):
    U, S, V = tch.svd(A)
    S_inv = (1./S).view(1,S.size(0))
    VS = V*S_inv # inverse of diagonal is just reciprocal of diagonal
    # print(U.shape)
    UtY = tch.mm(U.permute(1,0), b)
    return tch.mm(VS, UtY), None # The none here is just to mimic that lstsq has two outputs
