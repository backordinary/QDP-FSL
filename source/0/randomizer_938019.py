# https://github.com/armorsun/AmplifiQation/blob/32059a7e6648f5bc8994032b4664a2f45dfb2d5b/backend/randomizer.py
"""
Author: AmplifiQation

iQuHACK 2023
"""


import os
import qiskit as qt
from qiskit import QuantumCircuit, Aer
from qiskit import IBMQ
import covalent as ct


class RNG:
    """
    Random Number Generator using desired system.
    _backend:
        0 for IBM QC,
        1 for IBM Sim,
        2 for Aer Local Sim
    """
    backend: int

    def __init__(self, backend: int, token: str) -> None:
        """
        Initializes a random number generator
        :param backend: int
            User decides which system to use
        :param token: str
            IBM API to connect to QC
        :return: None:
        """
        IBMQ.save_account(token)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        if backend == 0:
            self.backend = provider.get_backend('ibm_nairobi')
        elif backend == 1:
            self.backend = provider.get_backend('simulator_stabilizer')
        else:
            self.backend = Aer.get_backend('qasm_simulator')

    @ct.electron
    def randomizer_circuit(self, num_qubits: int) -> int:
        """
        Returns a random integer based on the number of qubits available.
        :param num_qubits: int
            Number of qubits to use to get a random binary string
        :return: int:
            Random number
        """
        circuit = QuantumCircuit(num_qubits, num_qubits)
        for i in range(num_qubits):
            circuit.h(i)
        for i in range(num_qubits):
            circuit.measure(i, i)

        # circuit.draw(output='mpl')

        job = qt.execute(circuit, backend=self.backend, shots=1)
        counts = job.result().get_counts()
        return int(list(counts)[0], 2)
