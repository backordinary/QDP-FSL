# https://github.com/ChitraMadhaviVadlamani/VQE/blob/2092a00331f06bb42d2b34cbd34f37efc4b3c717/VQE/vqe4.py
#combined IX and XY

import numpy as np
from random import random
from scipy.optimize import minimize
#from qiskit.quantum_info.operators.operator import tensor
from qiskit import QuantumCircuit
#print(bottom^top,"@@") tensor product

from qiskit import QuantumCircuit
top = QuantumCircuit(1)
top.x(0)
bottom = QuantumCircuit(2)
bottom.cry(0.2, 0, 1)
tensored = bottom.tensor(top)
print(tensored.draw())

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

def hamiltonian_operator(a, b, c):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    H = aXY
    """
    #since no i the imaginary coefficients are set to 0
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": 1}, "label": "IX"},  #a is the weight and I is the Pauli Operator
                   {"coeff": {"imag": 0.0, "real": 1}, "label": "XY"},
                   {"coeff": {"imag": 0.0, "real": 1}, "label": "IZ"}
                   ]
    }
    #print(pauli_dict,"Paulis Dict")
    #print(WeightedPauliOperator.from_dict(pauli_dict),"WPO######")
    return WeightedPauliOperator.from_dict(pauli_dict)
    #return WeightedPauliOperator.from_dict(pauli_dict,"Pauli_dict 1")


scale = 10
a, b, c = (scale*random(), scale*random(), scale*random())
#print(a," ", b, " ", c, " ",d)
H = hamiltonian_operator(a, b, c) #method to make hamiltonians divided
#print("Here",H," This is the Hamiltonian 2",H.paulis)



#classical algorithm to find lowest energy state
exact_result = NumPyEigensolver(H).run()
reference_energy = min(np.real(exact_result.eigenvalues))
print('The exact ground state energy is: {}'.format(reference_energy))



#circuit preparation for anstaz
def quantum_state_preparation(circuit, circuit1, parameters):  #parameters change the radian value
    #todo: Try to return the radian i.e parameter of 0 and 1 for the state as well!!
    q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    q1 = circuit1.qregs[0]

    circuit.rx(parameters[0], q[0]) # q[0] is our one and only qubit XD
    circuit.ry(parameters[1], q[0])

    circuit1.rx(parameters[2], q1[0])
    circuit1.ry(parameters[3], q1[0])

    #print(circuit, "CIRCUIT with rx and ry q[0] parameters[0] and [1]: ", q[0],"..",parameters[0],"..",parameters[1])
    return circuit, circuit1


H_gate = U2Gate(0, np.pi).to_matrix()
print("H_gate:")
print((H_gate * np.sqrt(2)).round(5))

Y_gate = U2Gate(0, np.pi/2).to_matrix()
print("Y_gate:")
print((Y_gate * np.sqrt(2)).round(5))


def vqe_circuit(parameters, measure):
    print(measure,"VQE circuit")
    
    q = QuantumRegister(1)
    circuit = QuantumCircuit(q)
    q2 = QuantumRegister(1)
    c2 = ClassicalRegister(1)
    circuit1 = QuantumCircuit(q2)
    circuit, circuit1 = quantum_state_preparation(circuit, circuit1, parameters)
    
    if measure == 'XY':
        circuit.u2(0, np.pi/2, q[0])
        circuit1.u2(0, np.pi, q2[0])
        tensored = circuit1.tensor(circuit)
        tensored.measure_all(q[0],q2[0])
        print(tensored,"PLEASE WORK")
    
    elif measure == "IX":

        q3 = QuantumRegister(1)
        #circuit3 = QuantumCircuit(q3)
        #circuit3.id(q3[0])
        circuit1.u2(0, np.pi, q2[0])
        tensored = circuit1.tensor(circuit)
        #tensored = circuit1.tensor(circuit3)
        #tensored.measure_all(q[0],q2[0])
        tensored.measure_all(q[0],q3[0])
        print(tensored,"PLEASE  IX")

    elif measure == "IZ":

        q3 = QuantumRegister(1)
        circuit3 = QuantumCircuit(q3)
        circuit3.id(q3[0])
        tensored = circuit1.tensor(circuit)
        #tensored = circuit1.tensor(circuit3)
        #tensored.measure_all(q[0],q2[0])
        tensored.measure_all(q[0],q3[0])
        print(tensored,"PLEASE  IX")
    
    else:
        raise ValueError('Not valid input for measurement: input should be "IX" or "XY" or "IZ"')
    #print(circuit, "Circuit with measurement added")
    return tensored



def quantum_module(parameters, measure):
    print(measure,"Quantum Module")
    # measure
    if measure == 'IX':
        circuit = vqe_circuit(parameters, 'IX')
        #print(circuit, "here1")
    elif measure == 'XY':
        circuit = vqe_circuit(parameters, 'XY')
        #print(circuit, "here3")
    elif measure == 'IZ':
        circuit = vqe_circuit(parameters, 'IZ')
        #print(circuit, "here3")
    else:
        raise ValueError('Not valid input for measurement: input should be "IX" or "XY" or "IZ"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    #print(backend," backend")
    #print(circuit,"@@")
    job = execute(circuit, backend, shots=shots)
    #print(job, "job")
    result = job.result()
    #print(result, " result")
    counts = result.get_counts() #No of times we get 0 state and no of times we get 1 state
    #print(counts, "counts")
    
    # expectation value estimation from counts
    expectation_value = 0
    #print(counts,"outside")
    for measure_result in counts:
        #print(counts,"inside")
        #print(measure_result, " measure result ")
        sign = +1
        if measure_result == '01' or '10':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots  #first it gives probability of 0 adds it to expectation_value = 0; then it finds the prob 1 and then it adds it to prob 0 
        #print(expectation_value," EV")
        #print(sign, "@@",counts[measure_result],"!!")
        #print(expectation_value, " Expectation Value")
        
    return expectation_value


def pauli_operator_to_dict(pauli_operator):
    """
    from WeightedPauliOperator return a dict:
    {I: 0.7, X: 0.6, Z: 0.1, Y: 0.5}.
    :param pauli_operator: qiskit's WeightedPauliOperator
    :return: a dict in the desired form.
    """
    d = pauli_operator.to_dict()
    #print(pauli_operator, "Pauli operator, d = ", d)
    paulis = d['paulis']
    #print(paulis, "paulis")
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff
    return paulis_dict
pauli_dict = pauli_operator_to_dict(H)



def vqe(parameters):
        
    quantum_module_IX = pauli_dict['IX'] * quantum_module(parameters, 'IX')
    quantum_module_XY = pauli_dict['XY'] * quantum_module(parameters, 'XY')
    quantum_module_IZ = pauli_dict['IZ'] * quantum_module(parameters, 'IZ')
    print(quantum_module_IX," IXQM ",quantum_module_XY," XYQM")
    
    classical_adder = quantum_module_IX +  quantum_module_XY+ quantum_module_IZ
    return classical_adder

parameters_array = np.array([np.pi, np.pi, np.pi, np.pi])
tol = 1e-3 # tolerance for optimization precision.

vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
#print(vqe_result, " VQE Result")
print('The exact ground state energy is: {}'.format(reference_energy))
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))

print("The estimated ground state of molecule  is: q1= [",vqe_result.x[0],",",vqe_result.x[1],"] ","q2= [",vqe_result.x[2],",",vqe_result.x[3],"]")
