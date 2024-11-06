# https://github.com/marioantolinezh/QuantumComputing/blob/f884809837b738aa0a33b2f8a9fbee5fe9463d8b/QRNG.py
# QRNG from https://quantumcomputinguk.org/tutorials/16-qubit-random-number-generator
#Uses hadamard gates
#We are generating a 128 bit long number because AES uses 16 byte long numbers

#!pip install qiskit

# NOTE: Code taken from https://quantumcomputinguk.org/tutorials/16-qubit-random-number-generator
# Many thanks to them.
import qiskit
from qiskit.tools.jupyter import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute,IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import QasmSimulator

# Make sure Qiskit is working
#qiskit.__version__

# Use Aer's qasm_simulator
simulator = QasmSimulator()

q = QuantumRegister(128,'q')
c = ClassicalRegister(128,'c')
circuit = QuantumCircuit(q,c)
circuit.h(q) # Applies hadamard gate to all qubits
circuit.measure(q,c) # Measures all qubits 

# Draw the circuit
circuit.draw()

job = execute(circuit, simulator, shots=1)
                               
print('Executing Job...\n')                 
job_monitor(job)
counts = job.result().get_counts()

#print('RESULT: ',counts,'\n')
print('Press any key to close')
input()
