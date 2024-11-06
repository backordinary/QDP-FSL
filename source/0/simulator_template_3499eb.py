# https://github.com/kmesman/benchmarking/blob/60aa9d1ab032fa794d6a73529aedaf12b230b869/src/QPack_benchmark/simulator_template.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:49:51 2022

simulator template
Use this template to define the quantum simulator used for simulated results.
The Qiskit QASM simulator implemented here can be configured with noise.
The function name, input parameters and output result format should remain the
same in order to use it with the algorithm coordinator.
@author: koen
"""
from qiskit import QuantumCircuit, Aer, execute
from qiskit.test.mock import FakeVigo


def simulate_results(qasm_str, shots):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    noisy_backend = FakeVigo()
    backend = Aer.get_backend("qasm_simulator")

    # Execute the circuit and show the result.
    # job = execute(qc, noisy_backend, shots=shots)
    job = execute(qc, backend, shots=shots)

    result = job.result()
    return result.get_counts()
