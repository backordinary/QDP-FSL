# https://github.com/ccicconetti/qiskit-tests/blob/8e362a1d041b071b64b9630201e0bb56246c735a/tutorial/tutorial5.py
"""Deutsch example"""

import sys
import random
import numpy as np

# import basic plot tools
import matplotlib.pyplot as plt

# importing Qiskit
from qiskit.compiler import transpile
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel
from qiskit import(
    IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer)
from qiskit.tools.visualization import plot_histogram

def blackbox_c(a_qc):
    """Constant function"""
    #a_qc.iden(0)
    if random.random() > 0.5:
        print("Constant function: returns always 0")
        a_qc.iden(1)
    else:
        print("Constant function: returns always 1")
        a_qc.iden(1)

def blackbox_b(a_qc):
    """Balanced function"""
    print("Balanced function")
    a_qc.cx(0, 1)

# number of trials
# (does not apply to experiment_type == 'simulator-statevector')
shots = 1024

# one of:
#   simulator-statevector
#   simulator-qasm
#   simulator-noise
#   ibmq
experiment_type = 'simulator-noise'

# only meaningful with experiment_type == 'ibmq'
# or experiment_type == 'simulator-noise'
ibmq_backend = 'ibmq_essex'

# one of: constant, balanced
blackbox_type = 'constant'

if experiment_type == 'simulator-noise':
    print("Building circuit with noise from '{}'".format(ibmq_backend))

    # Build noise model from backend properties
    provider = IBMQ.load_account()
    backend = provider.get_backend(ibmq_backend)
    noise_model = NoiseModel.from_backend(backend)
    
    # Get coupling map from backend
    coupling_map = backend.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

# Create the Quantum Circuit
q = QuantumRegister(2, name='q')
c = ClassicalRegister(1, name='c')
qc = QuantumCircuit(q, c, name='qc')

# Prepare input
qc.h(0)
qc.x(1)
qc.h(1)
qc.barrier()

# Feed input to Uf
if blackbox_type == 'constant':
    blackbox_c(qc)
else:
    blackbox_b(qc)

# Retrieve measurement on qbit#0
qc.barrier()
qc.h(0)
#qc.h(1)

# Performance measurement
qc.barrier()
qc.measure(q[0], c[0])

# Print circuit
print(qc.draw(output='text'))

# Print input
# print(Statevector.from_instruction(qc).data)

if 'simulator' in experiment_type:
    print("Using simulator")

    # Use Aer's qasm_simulator
    if 'qasm' in experiment_type:
        result = execute(qc,
                         Aer.get_backend('qasm_simulator'),
                         shots=shots).result()

        print(result.get_counts())
        if '0' in result.get_counts():
            assert '1' not in result.get_counts()
            print("Oracle is constant")
        else:
            assert '0' not in result.get_counts()
            print("Oracle is balanced")

    elif 'statevector' in experiment_type:
        shots = 1
        result = execute(qc,
                         Aer.get_backend('statevector_simulator'),
                         shots=1).result()

        abs_values = abs(result.get_statevector(qc, decimals=2))
        print(abs_values)
        assert len(abs_values) == 4
        if abs_values[0] == 0.71 and abs_values[2] == 0.71:
            print("Oracle is constant")
        else:
            assert abs_values[1] == 0.71 and abs_values[3] == 0.71
            print("Oracle is balanced")

    elif 'noise' in experiment_type:
        result = execute(qc,
                         Aer.get_backend('qasm_simulator'),
                         coupling_map=coupling_map,
                         basis_gates=basis_gates,
                         noise_model=noise_model).result()
        counts = result.get_counts(0)
        print(counts)
        num0 = counts['0'] if '0' in counts else 0
        num1 = counts['1'] if '1' in counts else 0
        if num0 > num1:
            print("Oracle is constant")
        elif num0 < num1:
            print("Oracle is balanced")
        else:
            print("It's a tie!")

    else:
        raise Exception("Unknown experiment type: {}".format(experiment_type))


elif experiment_type == 'ibmq':
    print("Using IBMQ")

    # load IBM account
    provider = IBMQ.load_account()

    # Use IBM quantum computer
    backend = provider.get_backend(ibmq_backend)
    qc_compiled = transpile(qc, backend=backend, optimization_level=1)
    print(qc_compiled.draw(output='text'))

    job = execute(qc_compiled, backend=backend, shots=1024)
    job_monitor(job)
    results = job.result()
    answer = results.get_counts()

    threshold = int(0.01 * shots) # the threshold of plotting significant measurements
    filteredAnswer = {k: v for k,v in answer.items() if v >= threshold} # filter the answer for better view of plots

    removedCounts = np.sum([ v for k,v in answer.items() if v < threshold ]) # number of counts removed 
    filteredAnswer['other_bitstrings'] = removedCounts  # the removed counts are assigned to a new index

    plot_histogram(filteredAnswer)
    print(filteredAnswer)
    # executed on ibmq_essex gave answer:
    # {'0': 1010, '1': 14, 'other_bitstrings': 0.0}
    plt.show(block=True)

else:
    raise Exception("Invalid experiment type: {}".format(experiment_type))

sys.exit(0)