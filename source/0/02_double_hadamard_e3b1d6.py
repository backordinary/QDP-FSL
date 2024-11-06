# https://github.com/zpbappi/quantum-computing-readify-b2b-may2019/blob/0dc5c012afc58575727202de92f85aae1462ba3d/demos/02-double-hadamard.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)

# qc.x(q[0])
qc.h(q[0])
qc.h(q[0])

qc.measure(q, c)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend)
result = job.result()

print(result.get_counts(qc))
