# https://github.com/JorgeAGR/nmsu-course-work/blob/6cd204abbc074734fb7e8ca0e693a15e1cbe4ede/PHYS520/Project/project_vqls.py
'''
Based on the VQLS implementation in the Qiskit textbook.
https://qiskit.org/textbook/ch-paper-implementations/vqls.html
'''
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import numpy as np
from scipy.optimize import minimize

global shots
global backend
shots = 100000
backend = Aer.get_backend('qasm_simulator')

def hadamard_Test(gates, qubits, ancilla_index, parameters):

    qctl = QuantumRegister(2)
    qc = ClassicalRegister(1)
    circ = QuantumCircuit(qctl, qc)   
    
    circ.h(ancilla_index)
    
    # Fixed Ansatz
    circ.ry(parameters[0], qubits[0])

    # Data unitaries A_l
    if (gates[0] == 1):
        circ.cz(ancilla_index, qubits[0])
    elif (gates[0] == 2):
        circ.cx(ancilla_index, qubits[0])
    
    # Data unitaries A_lp
    if (gates[1] == 1):
        circ.cz(ancilla_index, qubits[0])
    elif (gates[1] == 2):
        circ.cx(ancilla_index, qubits[0])
    
    circ.h(ancilla_index)
    
    circ.measure(ancilla_index, 0)

    job = execute(circ, backend, shots=shots)

    result = job.result()
    outputstate = result.get_counts(circ)
    
    if ('1' in outputstate.keys()):
        m_sum = float(outputstate["1"])/shots
    else:
        m_sum = 0
    
    return 1 - 2*m_sum
    
def overlap_Hadamard_Test(gates, qubits, ancilla_index, parameters):

    qctl = QuantumRegister(3)
    qc = ClassicalRegister(1)
    circ = QuantumCircuit(qctl, qc)    

    circ.h(ancilla_index)
    
    # Ansatz on Qubit 1
    circ.ry(parameters[0], qubits[0])
    
    # U on Qubit 2
    circ.h(qubits[1])

    # Data unitaries A_l
    if (gates[0] == 1):
        circ.cz(ancilla_index, qubits[0])
    elif (gates[0] == 2):
        circ.cx(ancilla_index, qubits[0])
    
    # Data unitaries A_lp
    if (gates[1] == 1):
        circ.cz(ancilla_index, qubits[1])
    elif (gates[1] == 2):
        circ.cx(ancilla_index, qubits[1])
    
    # Overlap
    circ.cx(qubits[0], qubits[1])
    circ.h(qubits[0])
    circ.h(qubits[1])
    
    # Measure Ancilla
    circ.h(ancilla_index)
    circ.measure(ancilla_index, 0)

    job = execute(circ, backend, shots=shots)

    result = job.result()
    outputstate = result.get_counts(circ)
    
    if ('1' in outputstate.keys()):
        m_sum = float(outputstate["1"])/shots
    else:
        m_sum = 0
    
    return 1 - 2*m_sum

def cost_Function(parameters):

    psi_norm = 0

    for l in range(len(gates)):
        for lp in range(len(gates)):
            
            coeffs = coefficient_set[l]*coefficient_set[lp]

            beta = hadamard_Test([gates[l], gates[lp]], [1,], 0, parameters)

            psi_norm+=coeffs*beta

    b_psi = 0
    
    for l in range(len(gates)):
        for lp in range(len(gates)):

            coeffs = coefficient_set[l]*coefficient_set[lp]
            gamma = overlap_Hadamard_Test([gates[l], gates[lp]], [1, 2], 0, parameters)
        
            b_psi+=coeffs*gamma

    return 1-b_psi/psi_norm

A = np.array([[0.2, 0.25],
              [0.25, 0.2]])
b = np.array([[1,], [1]])
coefficient_set = [A[0,0], A[0,1]]
# 0 -> I, 1 -> Z, 2 -> X
gates = [0, 2]
x = np.linalg.inv(A) @ b
x_c = x/np.linalg.norm(x)
print('Classical Solution: {}'.format(x_c.tolist()))
print('Optimal Alpha: {:.4f}\n'.format(np.arccos(x_c[0,0])*2))

out = minimize(cost_Function, x0=[1 ,],
               method='COBYLA', options={'maxiter':100})
print(out)

out_f = out['x'].flatten()

qctl = QuantumRegister(1)
qc = ClassicalRegister(1)
circ = QuantumCircuit(qctl, qc)
circ.ry(out_f[0], 0)

backend = Aer.get_backend('statevector_simulator')

job = execute(circ, backend)

result = job.result()
x_q = result.get_statevector(circ, decimals=10).reshape((2,1))
print('Quantum Solution: {}'.format(x_q.tolist()))
print('Alpha Converged: {:.4f}'.format(out_f[0]))
