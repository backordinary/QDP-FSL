# https://github.com/andy0000-droid/Quantum/blob/14463e2b2d2de4c61276e44c6d3153664f5bfd1e/hackathon/717QEC.py
from qiskit import *
from qiskit.test.mock import FakeMontreal
device_backend = FakeMontreal()

qreg_dq = QuantumRegister(7, 'dq')
qreg_aq = QuantumRegister(6, 'aq')
#creg_dc = ClassicalRegister(7, 'dc')
creg_sc = ClassicalRegister(6, 'sc')
#circuit = QuantumCircuit(qreg_dq, qreg_aq, creg_dc, creg_sc)
circuit = QuantumCircuit(qreg_dq, qreg_aq, creg_sc)

circuit.reset(qreg_dq)
circuit.reset(qreg_aq)
circuit.barrier(qreg_dq[0], qreg_dq[1], qreg_dq[2], qreg_dq[3], qreg_dq[4], qreg_dq[5], qreg_dq[6])
circuit.barrier(qreg_aq[0], qreg_aq[1], qreg_aq[2], qreg_aq[3], qreg_aq[4], qreg_aq[5])
circuit.cx(qreg_dq[0], qreg_aq[0])
circuit.cx(qreg_dq[1], qreg_aq[0])
circuit.cx(qreg_dq[1], qreg_aq[1])
circuit.cx(qreg_dq[2], qreg_aq[1])
circuit.cx(qreg_dq[2], qreg_aq[2])
circuit.cx(qreg_dq[3], qreg_aq[2])
circuit.cx(qreg_dq[3], qreg_aq[3])
circuit.cx(qreg_dq[4], qreg_aq[3])
circuit.cx(qreg_dq[4], qreg_aq[4])
circuit.cx(qreg_dq[5], qreg_aq[4])
circuit.cx(qreg_dq[5], qreg_aq[5])
circuit.cx(qreg_dq[6], qreg_aq[5])
circuit.barrier(qreg_dq, qreg_aq)
#cx aq[0], dq[0];
#ccx aq[0], aq[1], dq[0];
#ccx aq[0], aq[1], dq[1];
#circuit.measure(qreg_dq, creg_dc)
circuit.measure(qreg_aq, creg_sc)
# @columns [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

print(circuit)

from qiskit.providers.aer import AerSimulator
sim_Montreal = AerSimulator.from_backend(device_backend)

# Simulating the circuit using the simulator to get the result
tcirc = transpile(circuit, sim_Montreal)
result_noise = sim_Montreal.run(tcirc).result()
counts = result_noise.get_counts(0)
print (counts)

for i in range(pow(2,6)):
    print(str(format(i,'b')).zfill(6),counts[str(format(i,'b')).zfill(6)])

print("\n")

for i in range(pow(2,6)):
    print(str(format(i,'b')).zfill(6), counts[str(format(i,'b')).zfill(6)]-(pow(2,10)/pow(2,c_circuit)))
