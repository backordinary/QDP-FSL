# https://github.com/andy0000-droid/Quantum/blob/ee467f7ebab7147ce17cdfc83e69b510e3f4ae2e/test.py
from qiskit import *
from qiskit.test.mock import FakeMontreal
device_backend = FakeMontreal()

q_circuit = 2
c_circuit = q_circuit

qreg_q = QuantumRegister(q_circuit, 'q')
creg_c = ClassicalRegister(c_circuit, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
print(circuit)

# Using Qiskit Aer's Qasm Simulator
# simulator = BasicAer.get_backend('qasm_simulator')
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
