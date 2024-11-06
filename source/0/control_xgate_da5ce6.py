# https://github.com/NeoTRAN001/QuamtumComputing/blob/262cd6713014957d46f50183306143c9c18cfcde/control_xgate.py
'''
CNOT gate
The controlled-NOT gate, also known as the controlled-x (CX) gate, 
acts on a pair of qubits, with one acting as ‘control’ and the other as 
‘target’. It performs a NOT on the target whenever the control is in state. 
If the control qubit is in a superposition, this gate creates entanglement.

All unitary circuits can be decomposed into single qubit gates and CNOT gates.
Because the two-qubit CNOT gate costs much more time to execute on real hardware than single qubit 
gates, circuit cost is sometimes measured in the number of CNOT gates.

For more information about the CNOT gate, see CXGate in the Qiskit Circuit 
Library.

'''


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.x(qreg_q[1])
circuit.x(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])