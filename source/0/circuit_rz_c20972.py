# https://github.com/achieveordie/HybridQNN/blob/9bc236216cfcf6eaff7c2d7becbcc930d161c7ba/circuits/circuit_rz.py
"""
Single Qubit where |+> is acted upon by RZ(theta) followed
by a z-measurement, only one parameter `theta`.
"""

import qiskit
import numpy as np
from main import HybridFunction
import torch


class Hybrid(torch.nn.Module):
    """ class to define the Hybrid quantum-classical layer"""

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.backend = backend
        self.shots = shots
        all_qubits = [i for i in range(n_qubits)]

        self._circuit.h(all_qubits)
        self._circuit.rz(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas):
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=[{self.theta: theta} for theta in thetas])
        counts = job.result().get_counts()
        values = np.array(list(counts.values()))
        states = np.array(list(counts.keys())).astype(float)
        probabilities = values / self.shots
        expectations = np.sum(probabilities * states)
        return np.array([expectations])
