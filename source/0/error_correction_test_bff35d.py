# https://github.com/andy0000-droid/Quantum/blob/e8b5f6942018892c4138695f7342eb8c7739428a/Error_correction_test.py
from qiskit import *

# Build quantum circuit
qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.reset(qreg_q[2])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.h(qreg_q[2])
circuit.h(qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.tdg(qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.t(qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.tdg(qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.t(qreg_q[1])
circuit.t(qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[2])
circuit.t(qreg_q[0])
circuit.tdg(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q,creg_c)

# Print quantum circuit
print(circuit)

# Using Qiskit Aer's Qasm Simulator
simulator = BasicAer.get_backend('qasm_simulator')

# Simulating the circuit using the simulator to get the result
job = execute(circuit, simulator)
result = job.result()

# Getting the aggregated binary outcomes of the circuit.
counts = result.get_counts(circuit)
print (counts)
for i in range(pow(2,3)):
    print(str(format(i,'b')).zfill(3),counts[str(format(i,'b')).zfill(3)])
print("\n")
