# https://github.com/andy0000-droid/Quantum/blob/6e6d500a26b7aebbd8cb53a1d19f932cd1b27959/Deutsch_Algorithm.py
from qiskit import *
from qiskit.test.mock import FakeMontreal
device_backend = FakeMontreal()

q_circuit = 3
c_circuit = 2

qreg_q = QuantumRegister(q_circuit, 'q')
creg_c = ClassicalRegister(c_circuit, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.reset(qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.x(qreg_q[2])
circuit.h(qreg_q[0])
circuit.h(qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.h(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.barrier(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.measure(qreg_q[0], creg_c[1])
circuit.measure(qreg_q[1], creg_c[0])
# @columns [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,5,6,7,7,8,8,8,9,9,10,11]


# Print quantum circuit
print(circuit)

# Using Qiskit Aer's Qasm Simulator
from qiskit.providers.aer import AerSimulator
simulator = AerSimulator.from_backend(device_backend)

# Simulating the circuit using the simulator to get the result
job = execute(circuit, simulator)
result = job.result()

# Getting the aggregated binary outcomes of the circuit.
counts = result.get_counts(circuit)
print (counts)

for i in range(pow(2,c_circuit)):
    print(str(format(i,'b')).zfill(2),counts[str(format(i,'b')).zfill(2)])

print("\n")

for i in range(pow(2,c_circuit)):
    print(str(format(i,'b')).zfill(2), counts[str(format(i,'b')).zfill(2)]-(pow(2,10)/pow(2,c_circuit)))
