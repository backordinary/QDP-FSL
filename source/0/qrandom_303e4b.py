# https://github.com/quantum-kittens/quantum-prompt-generator/blob/4bfc8178530d776dd4f0d84583b8203b13ca4871/promptgen/qrandom.py
import numpy as np
from qiskit import BasicAer, QuantumCircuit, execute


class QuantumRandomInt:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.num_qubits = max(1, int(np.ceil(np.log2(self.high - self.low))))
        self.circuit = self._make_circuit()
        self.randints = []

    def _make_circuit(self):
        circ = QuantumCircuit(self.num_qubits)
        circ.h(circ.qubits)
        circ.measure_all()
        return circ

    def _measure(self, size, backend=None):
        shots = max(size, 4096)
        if not backend:
            backend = BasicAer.get_backend("qasm_simulator")
        job = execute(self.circuit, backend=backend, shots=shots, memory=True)
        result = job.result().get_memory()
        return result

    def generate(self, size=1, backend=None):
        out = []
        while len(out) < size:
            if not self.randints:
                self.randints.extend(self._measure(size - len(out), backend))
            randint = self.low + int(self.randints.pop(), 2)
            while randint > self.high:
                randint = self.low + int(self.randints.pop(), 2)
            out.append(randint)

        return out


class QuantumDice(QuantumRandomInt):
    def __init__(self, high):
        super().__init__(1, high)

    def roll(self, times=1, modifier=0, backend=None):
        rolls = self.generate(times, backend)
        total = sum(rolls) + modifier
        return rolls, total
