# https://github.com/theDe-bugger/QxQ-QuantumQiskit/blob/17fa57af6bfc08da2a6420c91202ebda805e074a/random%20python%20notebooks/First.py
import qiskit
from qiskit import IBMQ
IBMQ.save_account('33ce7c5cd257ba06b8d1bd50a7b94561e0d472b1e566e1f2fe71e9a9f153060950e27d5cfb932246ffed817eeb9e1a47b2f5f9ab566cf04c921b7f5745735b45')
IBMQ.load_account()
from qiskit import *

# Creating Qubits
q = qk.QuantumRegister(2)
# Creating Classical Bits
c = qk.ClassicalRegister(2)

circuit = qk.QuantumCircuit(q, c)
# Hadamard Gate on the first Qubit
circuit.h(q[0])
# CNOT Gate on the first and second Qubits
circuit.cx(q[0], q[1])
# Measuring the Qubits
circuit.measure(q, c)

print(circuit)