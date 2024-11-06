# https://github.com/LeanderReascos/Quantum_Digital_Signatures/blob/7162a7fe3b57533476229382e72fe76139f9e265/libs/utils.py
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile
import numpy as np
from qiskit.test.mock import FakeAlmaden
from qiskit.providers.aer import AerSimulator
device_backend = FakeAlmaden()


def execute_circuit(qc, shots=1024, device=None):
    #device = Aer.get_backend('qasm_simulator') if device is None else device
    device = AerSimulator.from_backend(device_backend)
    transpiled_circuit = transpile(qc, device)
    counts = device.run(transpiled_circuit, shots=shots).result().get_counts()
    return counts

def basis_states_probs(counts):
    n = len(list(counts.keys())[0].replace(' ',''))
    keys = dict([(k.replace(' ',''),k) for k in list(counts.keys())])
    N = sum(list(counts.values()))
    return np.array([counts[keys[np.binary_repr(vals,n)]]/N if keys.get(np.binary_repr(vals,n)) is not None else 0 for vals in range(2**n)])

def swap_test(qc:QuantumCircuit,psi,phi,Ancila, ancila):
    qc.h(Ancila)
    [qc.cswap(Ancila,psi[i],phi[i]) for i in range(len(psi))]
    qc.h(Ancila)
    qc.measure(Ancila,ancila)
    return qc

def intracircuit_teleportation(qc:QuantumCircuit,Alice,Bob,alice):
    qc.cx(Alice,Bob)
    qc.h(Alice)
    qc.measure(Alice,alice)
    qc.z(Bob).c_if(alice,1)
    return qc