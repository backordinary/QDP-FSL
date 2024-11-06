# https://github.com/anott03/Quantum/blob/ad8dd086553cd374daa9abb7e40494f68647556f/foo.py
from qiskit import QuantumCircuit, Aer

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.h(1)
qc.cx(0, 2)
qc.cx(1, 2)

qc2 = QuantumCircuit(1, 1)
qc2.x(0)
#  qc.compose(qc2, qubits=[0])

qc.measure([0, 1, 2], [0, 1, 2])
sim = Aer.get_backend("aer_simulator")
job = sim.run(qc)
result = job.result()

print(result)
