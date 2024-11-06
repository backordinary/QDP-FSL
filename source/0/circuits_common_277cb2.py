# https://github.com/kingusiu/laspaclu/blob/4c93c464fd117009b05e3dc4c98939afb8be0c80/src/quantum/circuits_common.py
import numpy as np
from qiskit import QuantumCircuit


def swap_test(v1, v2):
    # basic swap test on amplitude encoded quantum states
    
    # create circuit with n qubits and one classical bit for measurement
    # qubits: first: ancilla, second: input vector, third: cluster center
    #(for latent space in R^2: 1 (ancilla) + 1 (x1,x2 coords of input) + 1 (x1,x2 coords of cluster))
    
    n = int(np.log2(len(v1))) if len(v1) > 1 else 1
    qc = QuantumCircuit(n*2+1, 1, name="swap_test")

    # control qubit default 0
    # append first vector
    qc.append(v1, range(1,n+1))
    # append second vector
    qc.append(v2, range(n+1,n*2+1))
    
    qc.barrier()
    # apply hadamard to control
    qc.h(0)
    # swap pairwise qubits controlled on ancilla
    for i in range(n):
        qc.cswap(0,i+1,i+n+1)
    # apply second hadamard
    qc.h(0)
    # measure control qubit
    qc.measure(0,0)

    return qc


def run_circuit(qc, shots=1024):
    backend = Aer.get_backend('qasm_simulator') # we choose the simulator as our backend
    counts = execute(qc, backend, shots=shots).result().get_counts() # we run the simulation and get the counts
    return counts
