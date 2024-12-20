# https://github.com/ctuning/ck-qiskit/blob/9a606a1ad9497d142486b788074827a3e6aeab11/program/qiskit-0.6-vqe/qiskit_vqe_common.py
#!/usr/bin/env python3

"""
This script runs Variational-Quantum-Eigensolver using Qiskit library

Example running it partially using CK infrastructure:
    ck virtual env  --tag_groups='compiler,python qiskit,lib vqe,utils vqe,hamiltonian deployed,ansatz deployed,optimizer' \
                    --shell_cmd="$HOME/CK/ck-qiskit/program/qiskit-vqe/qiskit_vqe_common.py --start_param_value=0.5"
"""

import os
import json
import time
import inspect

import numpy as np
from scipy import linalg as la

from qiskit import Aer, IBMQ, execute, QISKitError

from qiskit.tools.apps.optimization import make_Hamiltonian, group_paulis
from qiskit.tools.qi.pauli import Pauli, label_to_pauli
#from qiskit.tools.apps.optimization import eval_hamiltonian
from eval_hamiltonian import eval_hamiltonian

from vqe_utils import cmdline_parse_and_report, get_first_callable, NumpyEncoder

from vqe_hamiltonian import label_to_hamiltonian_coeff      # the file contents will be different depending on the plugin choice
import custom_ansatz                                        # the file contents will be different depending on the plugin choice

fun_evaluation_counter = 0    # global


def vqe_for_qiskit(q_device, sample_number, pauli_list, timeout_seconds, json_stream_file):

    def expectation_estimation(current_params, report):

        timestamp_before_ee = time.time()

        timestamp_before_q_run = timestamp_before_ee    # no point in taking consecutive timestamps

        ansatz_circuit  = ansatz_function(current_params)

        global fun_evaluation_counter
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 1, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 2, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 3, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 4, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 5, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 6, timeout_seconds))
        try:
            complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            #complex_energy, q_run_seconds = eval_hamiltonian(pauli_list_grouped, ansatz_circuit, sample_number, q_device)
            energy = complex_energy.real
            break
        except QISKitError as e:
            print("{}, trying again -- attempt number {}, timeout: {} seconds".format(e, 7, timeout_seconds))

        if len(q_run_seconds)>0:                            # got the real measured q time
            total_q_run_seconds = sum( q_run_seconds )
            measured            = 'remotely'
        else:                                               # have to assume
            total_q_run_seconds = time.time() - timestamp_before_q_run
            q_run_seconds       = [ total_q_run_seconds ]
            measured            = 'locally'

        q_runs              = len(q_run_seconds)
        total_q_run_shots   = sample_number * q_runs
        q_run_shots         = [sample_number] * q_runs

        report_this_iteration = {
            'total_q_seconds_per_c_iteration' : total_q_run_seconds,
            'seconds_per_individual_q_run' :    q_run_seconds,
            'total_q_shots_per_c_iteration' :   total_q_run_shots,
            'shots_per_individual_q_run' :      q_run_shots,
            'energy' :                          energy,
            'measured' :                        measured,
        }

        if report != 'TestMode':
            report['iterations'].append( report_this_iteration )
            report['total_q_seconds'] += report_this_iteration['total_q_seconds_per_c_iteration']  # total_q_time += total
            report['total_q_shots'] += report_this_iteration['total_q_shots_per_c_iteration']

            fun_evaluation_counter += 1

        report_this_iteration['total_seconds_per_c_iteration'] = time.time() - timestamp_before_ee

        print(report_this_iteration, "\n")
        json_stream_file.write( json.dumps(report_this_iteration, cls=NumpyEncoder)+"\n" )
        json_stream_file.flush()

        return energy

    # Groups a list of (coeff,Pauli) tuples into tensor product basis (tpb) sets
    pauli_list_grouped = group_paulis(pauli_list)


    report = { 'total_q_seconds': 0, 'total_q_shots':0, 'iterations' : [] }

    # Initial objective function value
    fun_initial = expectation_estimation(start_params, 'TestMode')
    print('Initial guess at start_params is: {:.4f}'.format(fun_initial))

    timestamp_before_optimizer = time.time()
    optimizer_output = minimizer_function(expectation_estimation, start_params, my_args=(report), my_options = minimizer_options)
    report['total_seconds'] = time.time() - timestamp_before_optimizer

    # Also generate and provide a validated function value at the optimal point
    fun_validated = expectation_estimation(optimizer_output['x'], 'TestMode')
    print('Validated value at solution is: {:.4f}'.format(fun_validated))

    # Exact (noiseless) calculation of the energy at the given point:
    complex_energy, _ = eval_hamiltonian(pauli_list, ansatz_function(optimizer_output['x']), 1, Aer.get_backend('statevector_simulator'))
    optimizer_output['fun_exact'] = complex_energy.real

    optimizer_output['fun_validated'] = fun_validated

    print('Total Q seconds = %f' % report['total_q_seconds'])
    print('Total Q shots = %d' % report['total_q_shots'])
    print('Total seconds = %f' % report['total_seconds'])

    return (optimizer_output, report)


if __name__ == '__main__':

    start_params, sample_number, q_device_name, minimizer_method, minimizer_options, minimizer_function = cmdline_parse_and_report(
        num_params                  = custom_ansatz.num_params,
        q_device_name_default       = 'qasm_simulator',
        q_device_name_help          = "Real devices: 'ibmqx4' or 'ibmqx5'. Use 'ibmq_qasm_simulator' for remote simulator or 'qasm_simulator' for local",
        minimizer_options_default   = '{"maxfev":200, "xatol": 0.001, "fatol": 0.001}',
        start_param_value_default   = 0.0
        )
    # q_device_name = os.environ.get('VQE_QUANTUM_BACKEND', 'qasm_simulator') # try 'qasm_simulator', 'ibmq_qasm_simulator', 'ibmqx4', 'ibmqx5'

    local_backends_names = [b.name() for b in Aer.backends(operational=True)]
    try:
        print('Trying to connect to the LOCAL backend "{}"...'.format(q_device_name))
        q_device = Aer.get_backend( q_device_name )
    except KeyError:
        print('Could not find the LOCAL q_device "{}" - available LOCAL q_devices:\n\t{}'.format(q_device_name, local_backends_names))
        try:
            api_token   = os.environ.get('CK_IBM_API_TOKEN')
            if not api_token:
                print('CK_IBM_API_TOKEN is not defined, so cannot connect to REMOTE q_devices - bailing out.'.format(q_device_name))
                exit(1)
            print('\nCK_IBM_API_TOKEN found.\nTrying to connect to the REMOTE q_device "{}"...'.format(q_device_name))
            IBMQ.enable_account( api_token )
            q_device = IBMQ.get_backend( q_device_name )
            remote_backends_names = [b.name() for b in IBMQ.backends(operational=True)]
        except KeyError as ex:
            print('Could not find the REMOTE q_device "{}" - available remote q_devices:\n\t{}'.format(q_device_name, remote_backends_names))
            exit(1)
    print('Using "{}" q_device...\n'.format(q_device.name()))

    # Ignore warnings due to chopping of small imaginary part of the energy
    #import warnings
    #warnings.filterwarnings('ignore')

    # Load the Hamiltonian into Qiskit-friendly format:
    pauli_list = [ [label_to_hamiltonian_coeff[label], label_to_pauli(label)] for label in label_to_hamiltonian_coeff ]

    # Calculate Exact Energy classically, to compare with quantum solution:
    #
    H = make_Hamiltonian(pauli_list)
    classical_energy = np.amin(la.eigh(H)[0])
    print('The exact ground state energy (the smallest eigenvalue of the Hamiltonian) is: {:.4f}'.format(classical_energy))

    # Load the ansatz function from the plug-in
    ansatz_method   = get_first_callable( custom_ansatz )
    ansatz_function = getattr(custom_ansatz, ansatz_method)     # ansatz_method is a string/name, ansatz_function is an imported callable

    timeout_seconds = int( os.environ.get('VQE_QUANTUM_TIMEOUT', '120') )

    json_stream_file = open('vqe_stream.json', 'a')

    # ---------------------------------------- run VQE: --------------------------------------------------

    (vqe_output, report) = vqe_for_qiskit(q_device, sample_number, pauli_list, timeout_seconds, json_stream_file)

    # ---------------------------------------- store the results: ----------------------------------------

    json_stream_file.write( '# Experiment finished\n' )
    json_stream_file.close()

    minimizer_src   = inspect.getsource( minimizer_function )
    ansatz_src      = inspect.getsource( ansatz_function )

    vqe_input = {
        "q_device_name"     : q_device_name,
        "minimizer_method"  : minimizer_method,
        "minimizer_src"     : minimizer_src,
        "minimizer_options" : minimizer_options,
        "ansatz_method"     : ansatz_method,
        "ansatz_src"        : ansatz_src,
        "sample_number"     : sample_number,
        "classical_energy"  : classical_energy
        }

    output_dict     = { "vqe_input" : vqe_input, "vqe_output" : vqe_output, "report" : report }
    formatted_json  = json.dumps(output_dict, cls=NumpyEncoder, sort_keys = True, indent = 4)

#    print(formatted_json)

    with open('ibm_vqe_report.json', 'w') as json_file:
        json_file.write( formatted_json )
