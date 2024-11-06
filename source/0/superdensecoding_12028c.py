# https://github.com/GabrielPontolillo/Testing_for_Quantum/blob/b62a1f1b2e30fa9c0089c3c3f584148c3bb11e4f/Algorithm%20Bases/Qiskit/SuperdenseCoding.py
import string

from AbstractCircuitGenerator import AbstractParametricCircuitGenerator
from qiskit import QuantumCircuit, Aer, execute


class SuperdenseCoding(AbstractParametricCircuitGenerator):
    """

    """
    def __init__(self):
        pass

    # def __init__(self, size: int = 2):
    #     self.size = size

    def generate_circuit(self, message: string):
        qc = self.create_bell_pair()
        qc = self.encode_message(qc, 1, message)
        qc = self.decode_message(qc)
        return qc

    @staticmethod
    def create_bell_pair():
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cx(1, 0)
        return qc

    @staticmethod
    def encode_message(qc, qubit, msg):
        if len(msg) != 2 or not set([0, 1]).issubset({0, 1}):
            raise ValueError(f"message '{msg}' is invalid")
        if msg[1] == "1":
            qc.x(qubit)
        if msg[0] == "1":
            qc.z(qubit)
        return qc

    @staticmethod
    def decode_message(qc):
        qc.cx(1, 0)
        qc.h(1)
        return qc


if __name__ == "__main__":
    print(SuperdenseCoding().generate_circuit("11"))
