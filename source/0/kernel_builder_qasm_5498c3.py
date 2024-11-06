# https://github.com/aide-qc/qcor/blob/8844bf8b7c1edfc916b040cb402d0d29171324a0/python/examples/kernel_builder_qasm.py
import qiskit
from qcor import * 

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1 ,2])

src = circ.qasm()

builder = KernelBuilder()
builder.from_qasm(src)
test_from_qasm = builder.create()

q = qalloc(3)
test_from_qasm(q)
print(q.counts())

builder = KernelBuilder()
builder.from_qiskit(circ)
test_from_qk = builder.create()

q = qalloc(3)
test_from_qk(q)
print(q.counts())
