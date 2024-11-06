# https://github.com/KartikChinda/QuantumComputing/blob/b6d50a244fb41d096cfda09baf4f2c3a8898b5b8/Grovers_algorithm.py
from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import quantumcircuit

oracle = QuantumCircuit(2, name='oracle')

# This flips the sign for the desired element.
oracle.cz(0, 1)
oracle.to_gate()

backend = Aer.get_backend('statevector_simulator')

# init grover circuit
grover_circ = quantumcircuit(2, 2)

# adding hadamard gate for the 2 qubits that we have, i.e. Superposition:
grover_circ.h([0, 1])

# adding the hadamard gates to the oracle.
grover_circ.append(oracle, [0, 1])

job = execute(grover_circ, backend)
result = job.result()

sv = result.get_statevector()

# THis is for storing the information in an array, using numpy.
np.around(sv, 2)

# Till here, we have made the oracle and flipped the state of the correct element. Now we will be implementing the second part of Grover's algo, which is the amplification.

reflection = quantumcircuit(2, name='reflection')

# This is for bringing the qubits back to the original state
reflection.h([0, 1])

# This is the implementation of the circuit shown in all of Grover's algorithms as the diffuser.
reflection.z([0, 1])
reflection.cz(0, 1)
reflection.h([0, 1])
reflection.to_gate()


# Adding everything and sending it again to the backend for simulation.
backend = Aer.get_backend('qasm_simulator')
grover_circ = QuantumCircuit(2, 2)
grover_circ.h([0, 1])
grover_circ.append(oracle, [0, 1])
grover_circ.append(reflection, [0, 1])
grover_circ.measure([0, 1], [0, 1])

# here, shots is going to be the number of calls we usually make to the backend, which will go highest to root n. Since this is just 2 qubits, we can run it one time.

job = execute(grover_circ, backend, shots=1)
result = job.result()
result.get_counts()
