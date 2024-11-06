# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/Shors/Shors.py
from QuantumAlgorithm import QuantumAlgorithm
import qiskit as q
from qiskit.converters import circuit_to_gate
import numpy as np


class Shors(QuantumAlgorithm):
    def __init__(self, args):
        super().__init__("Shors", args)

    def get_circuit(self):
        pass
