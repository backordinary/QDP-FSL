# https://github.com/achieveordie/HybridQNN/blob/9bc236216cfcf6eaff7c2d7becbcc930d161c7ba/circuits/circuit_cry.py
"""
A Two-Qubit circuit with a single parameter `theta`, where
they are prepared into |++> states followed by a Controlled-RY
Gate where the first qubit acts as the control and second
qubit is the target where RY(theta) is applied, only the second
qubit is measured in the z-basis
"""

import numpy as np
import qiskit
import torch
from main import HybridFunction


class Hybrid(torch.nn.Module):
    """ class to define the Hybrid quantum-classical layer"""

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(2, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


class QuantumCircuit:
    """ This class is to function the circuit"""
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits, 1)
        self.theta = qiskit.circuit.Parameter('theta')
        self.backend = backend
        self.shots = shots
        all_qubits = [i for i in range(n_qubits)]

        self._circuit.h(all_qubits)
        self._circuit.cry(self.theta, 0, 1)
        self._circuit.measure([1], [0])

    def run(self, thetas):
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=[{self.theta: theta} for theta in thetas])
        counts = job.result().get_counts()
        values = np.array(list(counts.values()))
        states = np.array(list(counts.keys())).astype(float)
        probabilities = values / self.shots
        expectations = np.sum(probabilities * states)
        return np.array([expectations])
