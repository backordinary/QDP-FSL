# https://github.com/avkhadiev/interpolators/blob/3911f7f5f03e9a5e2c5bd05d278a73fcb2cbe59a/vqe.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Used for testing: creates a VQE simulation object

import numpy as np
from qiskit import *

# local import
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states import Zero, VarFormBased
from qiskit.aqua.operators.expectations import PauliExpectation

from hamiltonian import h_even_op, h_odd_op
from schwinger_ansatz import SchwingerAnsatz
from optimizer import optimizer
from callbacks import simple_callback

# create initial state
num_qubits = 2
num_var_params = 3
var_form_depth = 1
initial_params = [0.0, 0.0, 0.0]
schwinger_form = SchwingerAnsatz(num_qubits, var_form_depth)
var_form_wavefunction = VarFormBased(schwinger_form, initial_params)

# create simulation instance
initial_point = initial_params
group_paulis = True       # group paulis into simultenously diagonalizable groups?
expectation = PauliExpectation(group_paulis)
max_evals_grouped = 1     # max number of evaluations performed simultaneously
aux_operators = None
quantum_instance = None


sim = VQE(h_even_op, schwinger_form, optimizer,
          initial_point,
          expectation,
          max_evals_grouped,
          aux_operators,
          simple_callback,
          quantum_instance)

if __name__ == '__main__':
    initial_params = [0.0, 0.0, 0.0]
    sim_circs = sim.construct_circuit(initial_params)
    print(sim_circs.__str__())
