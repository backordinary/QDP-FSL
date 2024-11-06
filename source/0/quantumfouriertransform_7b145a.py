# https://github.com/GabrielPontolillo/Quantum_Algorithm_Implementations/blob/3858b05392ec963e0d3703072823bed558abe424/Quantum%20Fourier%20Transform/Mutation%20Testing/QFT%20Mutation%20Testing%20Qiskit/QuantumFourierTransform.py
import warnings
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer, transpile
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram, plot_bloch_multivector

warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np

pi = np.pi

def flip_endian(dict):
    newdict = {}
    for key in list(dict):
        newdict[key[::-1]] = dict.pop(key)
    return newdict

def qft_rotations(circuit, n):
    #if qubit amount is 0, then do nothing and return
    if n == 0:
        #set it to measure the x axis
        #set_measure_x(circuit, 4)
        #circuit.measure_all()
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    return qft_rotations(circuit, n)

def qft_rotations_not_inplace(n):
    circuit = QuantumCircuit(n)
    #if qubit amount is 0, then do nothing and return
    if n == 0:
        #set it to measure the x axis
        #set_measure_x(circuit, 4)
        #circuit.measure_all()
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    return qft_rotations(circuit, n)
