# https://github.com/StevenSchuerstedt/QuantumComputing/blob/0b32d1c642450aec87b2fa0204ba285136875e6b/code/half_adder.py
### Half Adder using Qiskit
###########################
from qiskit import QuantumCircuit, Aer

sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(4, 2)

qc.x(0)
qc.x(1)

qc.barrier()

qc.cx(0, 2)
qc.cx(1, 2)
qc.ccx(0, 1, 3)

qc.barrier()

qc.measure(2, 0)
qc.measure(3, 1)

print(qc)

result = sim.run(qc).result()
counts = result.get_counts()
print(counts)
print("hello world!")
