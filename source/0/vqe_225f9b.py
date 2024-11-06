# https://github.com/Narip2/Quantum-Algorithm/blob/225d3c7e40598d27e650bc5a24b1fbf3eacc4314/VQE.py
from random import random

import numpy as np
import qiskit
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.aqua.operators import CircuitOp, WeightedPauliOperator
from scipy.optimize import minimize
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info import Operator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

scale = 10
a = random()*scale
b = random()*scale
c = random()*scale
d = random()*scale
# yy = Operator([
#     [1/np.sqrt(2),1/np.sqrt(2)],
#     [complex(0,1)/np.sqrt(2),complex(0,-1)/np.sqrt(2)]
# ])

yy = Operator([
    [1/np.sqrt(2),complex(0,-1)/np.sqrt(2)],
    [1/np.sqrt(2),complex(0,1)/np.sqrt(2)]
])

# qc.hamiltonian(0)

#the standard result(classical) for comparison
def hamiltonian_operator(a, b, c, d):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum
    that will be our Hamiltonian operator.

    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "I"},
                   {"coeff": {"imag": 0.0, "real": b}, "label": "Z"},
                   {"coeff": {"imag": 0.0, "real": c}, "label": "X"},
                   {"coeff": {"imag": 0.0, "real": d}, "label": "Y"}
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)
scale = 10
a, b, c, d = (scale*random(), scale*random(),
              scale*random(), scale*random())
H = hamiltonian_operator(a, b, c, d)
H = hamiltonian_operator(a, b, c, d)
exact_result = NumPyEigensolver(H).run()
reference_energy = min(np.real(exact_result.eigenvalues))
print('The exact ground state energy is: {}'.format(reference_energy))

def U_theta(circuit, parameters):
    circuit.rx(parameters[0],0)
    circuit.ry(parameters[1],0)

def measure(parameters, measure):

    q = QuantumRegister(1)
    classical = ClassicalRegister(1)
    qc = QuantumCircuit(q,classical)
    U_theta(qc,parameters)
    action = a

    if measure == 'X':
        qc.h(0)
        action = b
    elif measure == 'Y':
        # qc.U2Gate(0, np.pi/2)
        qc.unitary(yy, 0, label='yy')
        action = c

    elif measure == 'Z':
        action = d
    elif measure == 'I':
        return a

    qc.measure(0,0)
    # qc.draw(output='mpl')
    # plt.show()
    backend = Aer.get_backend("qasm_simulator")
    shots = 8192
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()

    if('0' in answer):
        res = action*(answer['0'] - (shots - answer['0']))/shots
    else:
        res = -1
    return res

def vqe(parameters):
    adder_res = measure(parameters,'I') + measure(parameters,'X') + measure(parameters, 'Y') + measure(parameters, 'Z')
    return adder_res





parameters = [np.pi, np.pi]
tol = 1e-3 # tolerance for optimization precision.
vqe_result = minimize(vqe, parameters, method="Powell", tol=tol)
print(vqe_result.fun)



# backend = Aer.get_backend("qasm_simulator")
# shots = 1000
# results = execute(qc, backend=backend, shots=shots).result()
# answer = results.get_counts()
# print(answer)
# plot_histogram(answer)
# plt.show()

