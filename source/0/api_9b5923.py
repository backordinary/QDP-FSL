# https://github.com/ryomtk/QuantumWalkOnUnity/blob/0435c018dfa3837d4b8a4db57249fe6f2d67f671/FlaskServer/api.py
#!/usr/bin/env python3
from quantum_walker import ibmsim, getQWC ,resetQWC
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import quantumregister

def run_qasm(qasm, backend_to_run="qasm_simulator"):
    qc = QuantumCircuit.from_qasm_str(qasm)
    backend = Aer.get_backend(backend_to_run)
    job_sim = execute(qc, backend)
    sim_result = job_sim.result()
    return sim_result.get_counts(qc)


def run_walk(times,repeat,n,shots):

    quantumWalkCircuit = getQWC(times,repeat,n)

    result = ibmsim(quantumWalkCircuit,shots)

    return result

def reset_cat(n):
    resetQWC(n)