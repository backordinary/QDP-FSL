# https://github.com/adamcallison/cpqaoa/blob/676587e0d1e18ae80c4d6aa86385a875e3a7f9bd/mix_util.py
import numpy as np
from functools import lru_cache

from qiskit import QuantumCircuit, QuantumRegister, transpile

@lru_cache(maxsize=1)
def standard_mixer_eigenvalues(n):
    # returns diagonal of standard mixer that has been hadamard Hd_transformed
    # into Z eigenbasis
    N = 2**n
    eigvals = np.zeros(N)
    for j in range(N):
        jstr = bin(j)[2:]
        ones = jstr.count('1')
        eigvals[j] = 2*ones
    return eigvals

def standard_mixer_circuit(n, param, qubits=None):
    if qubits is None:
        qubits = range(n)
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)
    for q1 in qubits:
        qc.rx(-2*param, q1)
    qc.global_phase = -(param*n)
    return qc
