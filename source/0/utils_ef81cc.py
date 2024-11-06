# https://github.com/Allenator/nusynth/blob/17e2857747dc4329dfc5e04823fe24418adebd64/nusynth/utils.py
import contextlib
import math

import joblib
import numpy as np
from qiskit import Aer
from scipy.stats import unitary_group
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# https://scicomp.stackexchange.com/questions/10748/cartesian-products-in-numpy
def cross_product(x, y):
    cross_product = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return cross_product


def random_unitary(n_qubits, size=1, as_vec=False):
    u = unitary_group.rvs(2 ** n_qubits, size=size)

    if as_vec:
        return unitary_to_vec(u)
    else:
        return u


def unitary_to_vec(unitary):
    s = unitary.shape
    assert s[-1] == s[-2]

    unitary = unitary.reshape(-1, s[-1] * s[-2]).squeeze()

    return np.concatenate((unitary.real, unitary.imag), axis=-1)


def vec_to_unitary(vec):
    s = vec.shape
    n_c = int(s[-1] / 2)
    dim = int(math.sqrt(n_c))
    assert dim ** 2 == n_c

    vec = vec.reshape(-1, 2, n_c)
    unitary = np.apply_along_axis(lambda args: [complex(*args)], -2, vec)

    return unitary.reshape(-1, dim, dim).squeeze()


def circuit_to_unitary(circuit):
    res = Aer.get_backend('unitary_simulator').run(circuit).result()
    return res.get_unitary(circuit)


def pca(main, aux_list, n_components=2):
    ss = StandardScaler()
    pca = PCA(n_components=n_components)
    main_r = ss.fit_transform(main)
    main_pcs = pca.fit_transform(main_r)

    aux_pcs_list = [
        pca.transform(ss.transform(aux))
        for aux in aux_list
    ]

    return main_pcs, aux_pcs_list
