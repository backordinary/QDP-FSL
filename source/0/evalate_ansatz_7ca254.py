# https://github.com/avkhadiev/interpolators/blob/3911f7f5f03e9a5e2c5bd05d278a73fcb2cbe59a/evalate_ansatz.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reads in VQE parameters & calculates exactly the unitary matrix that
# corresponds to the variational circuits at those values. Saves in file.

import numpy as np
import logging

from qiskit import *
from scipy.io import mmwrite

from qiskit.aqua.operators import CircuitOp
from schwinger_ansatz import SchwingerAnsatz
from name_conventions import opt_params_fname

# setup logging
FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT, level=logging.ERROR)

sim_name = 'h_even'
vqs_circ_mat_fname = "./data_L2/circuitMatrix_L2.mtx"

if __name__ == '__main__':
    print("Loading variational parameters...")
    opt_params = np.loadtxt(opt_params_fname(sim_name))
    var_form_depth = int(len(opt_params)/3)
    print("Creating a variational form...")
    num_qubits = 3
    var_form = SchwingerAnsatz(num_qubits, var_form_depth)
    print("Building a circuit at variational parameters...")
    vqs_circ = var_form.construct_circuit(opt_params)
    print("Evaluating the corresponding matrix...")
    vqs_circ_mat = CircuitOp(vqs_circ).to_matrix()
    print("Saving matrix to output file %s" % (vqs_circ_mat_fname, ))
    mmwrite(vqs_circ_mat_fname, vqs_circ_mat)

