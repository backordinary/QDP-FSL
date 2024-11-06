# https://github.com/ChitraMadhaviVadlamani/VQE/blob/2092a00331f06bb42d2b34cbd34f37efc4b3c717/VQE/sir.py
import numpy as np
from random import random
from scipy.optimize import minimize

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

def quantum_state_preparation(circuit, parameters):
    q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    circuit.rx(parameters[0], q[0]) 
    circuit.ry(parameters[1], q[0])
    circuit.rx(parameters[2], q[1]) 
    circuit.ry(parameters[3], q[1])
    return circuit

def vqe_circuit(parameters, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    if measure == 'Z':
        circuit.measure(q[0], c[0])
    elif measure == 'X':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'Y':
        circuit.u2(0, np.pi/2, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'IX':
        circuit.measure(q[0], c[0])
        circuit.u2(0, np.pi, q[1])
        circuit.measure(q[1], c[1])
    elif measure == 'IZ':
        circuit.measure(q[0], c[0])
        circuit.u2(0, np.pi/2, q[1])
        circuit.measure(q[1], c[1])
    elif measure == 'XY':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0], c[0])
        circuit.u2(0, np.pi/2, q[1])
        circuit.measure(q[1], c[1])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')
    print(circuit)
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
    elif measure == 'IX':
        circuit = vqe_circuit(parameters, 'IX')
    elif measure == 'IZ':
        circuit = vqe_circuit(parameters, 'IZ')
    elif measure == 'XY':
        circuit = vqe_circuit(parameters, 'XY')
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    #print("result : {} and counts : {}".format(result, counts))
    
    # expectation value estimation from counts
    expectation_value = 0
    #print(counts)
    for measure_result in counts:
        sign = +1
        if (measure_result == '01' or measure_result == '10'):
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
        
    return expectation_value

def vqe(parameters):
        
    # quantum_modules
    #quantum_module_IX = pauli_dict['IX'] * quantum_module(parameters, 'IX')
    quantum_module_IX = 0
    #quantum_module_IZ = pauli_dict['IZ'] * quantum_module(parameters, 'IZ')
    quantum_module_IZ=0
    quantum_module_XY = pauli_dict['XY'] * quantum_module(parameters, 'XY')
    
    # summing the measurement results
    classical_adder = quantum_module_IX + quantum_module_IZ + quantum_module_XY

    return classical_adder

def hamiltonian_operator_two_qubits(a, b, c, d):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    
    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": 0}, "label": "IX"},
                   {"coeff": {"imag": 0.0, "real": 0}, "label": "IZ"},
                   {"coeff": {"imag": 0.0, "real": 1}, "label": "XY"},
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)

scale = 10
a, b, c, d = (scale*random(), scale*random(), 
              scale*random(), scale*random())
H_two_qubits = hamiltonian_operator_two_qubits(a, b, c, d)

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
pauli_dict = pauli_operator_to_dict(H_two_qubits)

exact_result = NumPyEigensolver(H_two_qubits).run()
reference_energy = min(np.real(exact_result.eigenvalues))
print('The exact ground state energy is: {}'.format(reference_energy))

parameters_array = np.array([np.pi, np.pi, np.pi, np.pi])
tol = 1e-3 # tolerance for optimization precision.

vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))