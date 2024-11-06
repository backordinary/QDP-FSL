# https://github.com/npgh2009/VQE-CQSim/blob/9bee775bba69a8309ee4f3ac8a875f32913c3948/Huy/singleQubit.py
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:22:40 2020

@author: lul
"""


import numpy as np
from random import random
from scipy.optimize import minimize

import qiskit
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver


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

exact_result = NumPyEigensolver(H).run()
reference_energy = min(np.real(exact_result.eigenvalues))
print('The exact ground state energy is: {}'.format(reference_energy))


def quantum_state_preparation(circuit, parameters):
    q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    circuit.rx(parameters[0], q[0]) # q[0] is our one and only qubit XD
    circuit.ry(parameters[1], q[0])
    return circuit

"""
H_gate = U2Gate(0, np.pi).to_matrix()
print("H_gate:")
print((H_gate * np.sqrt(2)).round(5))

Y_gate = U2Gate(0, np.pi/2).to_matrix()
print("Y_gate:")
print((Y_gate * np.sqrt(2)).round(5))
"""


def vqe_circuit(parameters, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = qiskit.QuantumRegister(1)
    c = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    if measure == 'Z':
        circuit.measure(q[0], c[0])
    elif measure == 'X':
        circuit.u(np.pi/2, 0, np.pi, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'Y':
        circuit.u(np.pi/2, 0, np.pi/2, q[0])
        circuit.measure(q[0], c[0])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

    return circuit


def quantum_module(parameters, measure):
    # measure
    if measure == 'I':
        return 1
    elif measure == 'Z':
        circuit = vqe_circuit(parameters, 'Z')
    elif measure == 'X':
        circuit = vqe_circuit(parameters, 'X')
    elif measure == 'Y':
        circuit = vqe_circuit(parameters, 'Y')
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')
    
    shots = 8192
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        sign = +1
        if measure_result == '1':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
        
    return expectation_value


def pauli_operator_to_dict(pauli_operator):
    """
    from WeightedPauliOperator return a dict:
    {I: 0.7, X: 0.6, Z: 0.1, Y: 0.5}.
    :param pauli_operator: qiskit's WeightedPauliOperator
    :return: a dict in the desired form.
    """
    d = pauli_operator.to_dict()
    paulis = d['paulis']
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff

    return paulis_dict
pauli_dict = pauli_operator_to_dict(H)


def vqe(parameters):
        
    # quantum_modules
    quantum_module_I = pauli_dict['I'] * quantum_module(parameters, 'I')
    quantum_module_Z = pauli_dict['Z'] * quantum_module(parameters, 'Z')
    quantum_module_X = pauli_dict['X'] * quantum_module(parameters, 'X')
    quantum_module_Y = pauli_dict['Y'] * quantum_module(parameters, 'Y')
    
    # summing the measurement results
    classical_adder = quantum_module_I + quantum_module_Z + quantum_module_X + quantum_module_Y
    
    return classical_adder

parameters_array = np.array([np.pi, np.pi])
tol = 1e-3 # tolerance for optimization precision.

vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
#print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))