# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/basic-Copy2-checkpoint.py
import numpy as np
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, execute, QuantumCircuit
from qiskit.circuit.library import QuantumVolume


qc = QuantumCircuit(7)

qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)
qc.h(5)

qc.x(0)
qc.x(2)
qc.x(3)

qc.mct([0,1,2,3,4,5],6)
qc.z(6)
qc.mct([0,1,2,3,4,5],6)

qc.x(0)
qc.x(2)
qc.x(3)

qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)
qc.h(5)

qc.x(0)
qc.x(1)
qc.x(2)
qc.x(3)
qc.x(4)
qc.x(5)

qc.h(5)

qc.mct([0,1,2,3,4],5)

qc.h(5)

qc.x(0)
qc.x(1)
qc.x(2)
qc.x(3)
qc.x(4)
qc.x(5)

qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.h(4)
qc.h(5)

qc.measure_all()

sim = AerSimulator(method="statevector")

#qc_compiled = transpile(circ, backend)
#job = backend.run(qc_compiled, shots=10, blocking_enable=True, blocking_qubits=1)
#result = job.result()

circ = transpile(qc)

result = execute(circ, sim, shots=10, blocking_enable=True, blocking_qubits=5).result()


counts = result.get_counts(circ)

print(counts)
#print(result)