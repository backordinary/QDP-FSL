# https://github.com/unimauro/qiskit-summer-2020/blob/21329ac62bf9922fa6ccb391a56c84a033429020/Laboratorio%201/ex1.py
def initialize_qubit(given_circuit, qubit_index):
    
    import numpy as np
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    initial_state=[np.sqrt(0.7),np.sqrt(0.30)]
    given_circuit.initialize(initial_state,qubit_index)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return given_circuit

def entangle_qubits(given_circuit, qubit_Alice, qubit_Bob):
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    given_circuit.h(qubit_Alice)
    given_circuit.cx(qubit_Alice,qubit_Bob)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    
    return given_circuit

def bell_meas_Alice_qubits(given_circuit, qubit1_Alice, qubit2_Alice, clbit1_Alice, clbit2_Alice):
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    given_circuit.cx(qubit1_Alice,qubit2_Alice)
    given_circuit.h(qubit1_Alice)
    mycircuit.measure([qubit1_Alice, qubit2_Alice],[clbit1_Alice[0], clbit2_Alice[0]])
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    return given_circuit

def controlled_ops_Bob_qubit(given_circuit, qubit_Bob, clbit1_Alice, clbit2_Alice):
    
    ### WRITE YOUR CODE BETWEEN THESE LINES - START
    given_circuit.x(qubit_Bob).c_if(clbit2_Alice,1)
    given_circuit.z(qubit_Bob).c_if(clbit1_Alice,1)
    ### WRITE YOUR CODE BETWEEN THESE LINES - END
    
    return given_circuit

from qiskit import QuantumRegister, ClassicalRegister,QuantumCircuit

### set up the qubits and classical bits
all_qubits_Alice = QuantumRegister(2)
all_qubits_Bob = QuantumRegister(1)
creg1_Alice = ClassicalRegister(1)
creg2_Alice = ClassicalRegister(1)

### quantum teleportation circuit here
# Initialize
mycircuit = QuantumCircuit(all_qubits_Alice, all_qubits_Bob, creg1_Alice, creg2_Alice)
initialize_qubit(mycircuit, 0)
mycircuit.barrier()
# Entangle
entangle_qubits(mycircuit, 1, 2)
mycircuit.barrier()
# Do a Bell measurement
bell_meas_Alice_qubits(mycircuit, all_qubits_Alice[0], all_qubits_Alice[1], creg1_Alice, creg2_Alice)
mycircuit.barrier()
# Apply classically controlled quantum gates
controlled_ops_Bob_qubit(mycircuit, all_qubits_Bob[0], creg1_Alice, creg2_Alice)

### Look at the complete circuit
print(mycircuit.draw(output='text'))

### store the circuit as the submitted answer
answer = mycircuit