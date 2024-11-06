# https://github.com/avkhadiev/interpolators/blob/3911f7f5f03e9a5e2c5bd05d278a73fcb2cbe59a/run_vqe.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# runs VQE for a given L=2 hamiltonian on 2 qubits

import numpy as np
import time
import argparse
import logging

from qiskit import *

from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.initial_states import Zero, VarFormBased
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel

from hamiltonian import get_matrix, pauli_operator, h_even_fname, h_odd_fname, subtract_trace
from schwinger_ansatz import SchwingerAnsatz
from name_conventions import opt_params_fname, vqe_log_fname
from callbacks import simple_callback, analyze_optimizer

# setup logging
FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT, level=logging.ERROR)

# variational params
num_var_params = 3
var_form_depth = 1
initial_params = [0.0, 0.0, 0.0]

# optimization parameters
max_trials = 500
optimizer = qiskit.aqua.components.optimizers.SPSA(max_trials)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Run VQE for a given L=2 Hamiltonian on 2 qubits'
        )
    # VQE params
    parser.add_argument('--name', type=str, default = 'test')
    parser.add_argument('--h_fname', type=str, default = h_even_fname)
    parser.add_argument('--nshots', type=int, default = 1024)
    parser.add_argument('--backend', type=str, default = 'qasm_simulator')
    parser.add_argument('--noise', type=bool, default = False)
    parser.add_argument('--max_trials', type=int, default = 500)
    # parse & log arguments
    args = parser.parse_args()
    logging.info(f'args = {args}')
    sim_name = args.name
    # get matrix, remove trace
    h_mat = get_matrix(args.h_fname)
    h_mat_traceless, h_mat_trace = subtract_trace(h_mat)
    h_operator = pauli_operator( h_mat_traceless )
    if (args.backend == 'qasm_simulator'):
        backend_sim = Aer.get_backend(args.backend)
    else:
        err_msg = "can't yet handle backend other than qasm_simulator"
        logging.error(err_msg)
        raise ValueError(err_msg)
    # create variational form and optimizer instance
    num_qubits = 2
    schwinger_form = SchwingerAnsatz(num_qubits, var_form_depth)
    optimizer = qiskit.aqua.components.optimizers.SPSA(args.max_trials)
    # load IBMQ account and create quantum instance
    provider=IBMQ.load_account()
    # configure noise model
    noise_backend = None
    basis_gates = None
    coupling_map = None
    noise_model = None
    if (args.backend=='qasm_simulator' and args.noise):
        print("Using a noise model...")
        noise_backend = provider.get_backend('ibmqx2')
        noise_model = NoiseModel.from_backend(noise_backend)
        coupling_map = noise_backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
    # TODO add measurement error mitigation
    measurement_error_mitigation_cls=None
    cals_matrix_refresh_period=None
    measurement_error_mitigation_shots=None
    my_quantum_instance = QuantumInstance(backend_sim,
                                          shots=args.nshots,
                                          # noise model
                                          basis_gates=basis_gates,
                                          coupling_map=coupling_map,
                                          noise_model=noise_model,
                                          # error mitigation
                                          measurement_error_mitigation_cls=None,
                                          cals_matrix_refresh_period=None,
                                          measurement_error_mitigation_shots=None
                                          )
    # create simulation instance
    initial_point = initial_params
    group_paulis = True       # group paulis into simultenously diagonalizable groups?
    expectation = PauliExpectation(group_paulis)
    max_evals_grouped = 1     # max number of evaluations performed simultaneously
    include_custom = False
    aux_operators = None
    sim = VQE(h_operator,
              schwinger_form,
              optimizer = optimizer,
              initial_point = initial_point,
              expectation = expectation,
              # include_custom = include_custom,
              max_evals_grouped = max_evals_grouped,
              aux_operators = aux_operators,
              callback = simple_callback,
              quantum_instance = my_quantum_instance)
    # run
    start_time = time.time()
    res = sim.run(my_quantum_instance)
    logging.info(f'VQE run time {time.time()-start_time:2.2f}')
    print(res)
    opt_params = res['opt_params']
    print(opt_params)
    print(sim._energy_evaluation(opt_params))
    print(sim._energy_evaluation(opt_params))
    print(sim._energy_evaluation(opt_params))
    print(sim._energy_evaluation(opt_params))
    print(sim._energy_evaluation(opt_params))
    np.savetxt(opt_params_fname(args.name), np.array(sim._ret['opt_params']))

