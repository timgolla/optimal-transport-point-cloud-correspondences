# Author Tim Golla <tim.golla.official@gmail.com>

import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import warnings
import time


def sinkhorn_sparse(K, s, d, epsilon=1e-5, maxiter=1000, verbose=False, plot=False):
    st = time.time()
    s /= s.sum()
    d /= d.sum()
    u = np.ones(s.shape)
    v = np.ones(d.shape)
    assert(np.isclose(np.sum(s), 1))
    assert(np.isclose(np.sum(d), 1))
    difference_norm = 2*epsilon
    it = 0
    plotstep = 100
    T = K
    T_previous = K
    while difference_norm > epsilon and it < maxiter:
        try:
            u = s / K.dot(v)
            v = d / K.T.dot(u)
        except FloatingPointError:
            warnings.warn(
                "Floating point error. Using last working tranport matrix and returning")
            break
        if sp.issparse(K):
            T = (K.multiply(v[None, :])).multiply(u[:, None])
            difference_norm = spl.norm(T - T_previous)
            if not np.all(np.isfinite(T.data)):
                T = T_previous
                print("The transport matrix has nans. Using last version and breaking.")
                break
        else:
            T = u[:, None]*K*v[None, :]
            difference_norm = npl.norm(T - T_previous)
        T_previous = T
        if verbose:
            print("difference norm = " + str(difference_norm))
        if plot:
            if it % plotstep == 0:
                if sp.issparse(K):
                    im = plt.spy(T)
                else:
                    im = plt.imshow(T)
                plt.savefig("j:/temp/" + str(it) + ".png")
                plt.close()
        it += 1
    if verbose:
        print("Done. Constructing transport matrix")
    T_sum = T.sum()
    if not np.isclose(T_sum, 1):
        warnings.warn("Sum transport matrix (should be 1): " + str(T_sum))
        T /= T_sum
    if plot:
        if sp.issparse(K):
            plt.spy(T)
        else:
            plt.imshow(T)
        plt.savefig("TransportMatrix.png")
        plt.close()
    et = time.time()
    totaltime = et - st
    return T, totaltime
