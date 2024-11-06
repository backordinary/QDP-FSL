# https://github.com/TimVroomans/Quantum-Mastermind/blob/32a6f6cf8daa40faaa378a5fee0584a8469fc4c2/src/dies/ndie.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from experiment.qiskit_experiment import QiskitExperiment

experiment = QiskitExperiment()


def _die(n):
    num_qubits = np.ceil(np.log2(n))

    minroll = n
    # In case we roll above the desired die size, we roll again.
    while minroll >= n:
        minroll = _roll(num_qubits)

    return minroll + 1


def _roll(num_qubits):
    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_qubits)
    circuit = QuantumCircuit(q, c)

    # Add H gates to all qubits.
    circuit.h(q)
    # Measure all qubits.
    circuit.measure(q, c)

    # Run our circuit once via QI api.
    result = experiment.run(circuit, 1)

    # Collect our binary string (since the qubits are all measured we should have one outcome).
    binary_string = list(result.get_counts())[0]

    # Convert the text string to an actual number.
    return int(binary_string, 2)


class Die:
    def __init__(self, die_size):
        self.die_size = die_size

    def roll(self):
        return _die(self.die_size)
