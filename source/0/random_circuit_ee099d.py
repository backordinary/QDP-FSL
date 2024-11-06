# https://github.com/iuliazidaru/qiskit-runtime-demo/blob/9f295999ca616360d32aa7b2d8e3d9e918de203f/qiskit_runtime_demo/libcode/random_circuit.py
import random
from qiskit import transpile
from qiskit.circuit.random import random_circuit


class RandomCircuit:

    def __init__(self, iterations, backend):
        self.iterations = iterations
        self.backend = backend
        self.depth = 0

    def create_new_random_circuit(self):
        """Creates random circuits with increased circuit depth"""
        circuit = random_circuit(num_qubits=5, depth=self.depth, measure=True, seed=random.randint(0, 1000))
        self.depth = self.depth + 1
        return transpile(circuit, self.backend)