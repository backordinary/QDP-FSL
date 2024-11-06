# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/LiH.py
from qiskit import *
from qiskit.quantum_info import Kraus
from qiskit.providers.aer.noise import NoiseModel, pauli_error, QuantumError