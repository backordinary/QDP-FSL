# https://github.com/avkhadiev/interpolators/blob/3911f7f5f03e9a5e2c5bd05d278a73fcb2cbe59a/interpolators.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Defines a interpolating operators for VQE

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, identity
from scipy.sparse import block_diag

from qiskit import *
from qiskit.quantum_info import Pauli

# local imports
from qiskit.aqua.operators import MatrixOp, PauliOp, PrimitiveOp
from qiskit.aqua.operators import MatrixOperator, TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.expectations import PauliExpectation

from hamiltonian import get_matrix, pauli_operator
from name_conventions import data_dir_name

# specify parameters
# TODO have this read from a file that is shared with the Mathematica script too,
# so that params reside in a single place
J = 0.8333
w = 0.500
m = -0.750


def interp_op_fpath(index): # one-indexed
    # change as necessary
    dir_str = 'data_L2_J0p833_w0p500_m-0p750'
    fpath = "./%s/interpOp%d_L2_lmd1_1_lmd2_3.mtx" % (dir_str, index,)
    return fpath

def get_interp_op_list(nops):
    interp_ops = np.zeros(nops, dtype=object)
    for i in range(nops):
        interp_mat = get_matrix( interp_op_fpath(i+1) )
        interp_ops[i] = pauli_operator( interp_mat )
    return interp_ops

if __name__ == '__main__':
    print(interp_op_fpath(1))
