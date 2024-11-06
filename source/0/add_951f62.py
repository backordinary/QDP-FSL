# https://github.com/Sassa-nf/pi/blob/57fabb0c4957a1133790a30410b7b68348e0e2f2/quant/add.py
# The old problem: add two numbers
#
# a -*---*-----------
# b -|---|-*-*---*---
# c -*-*-|-|-|---|---
# d -|-|-|-*-|-*-|-*-
# 0 -|-x-x-|-|-|-|-|-----x-X-
# 0 -x-----|-*-*-x-x---x-|-X-
# 0 -------x-x-x-----x-|-|-X-
# 1 -----------------*-*-*---

from qiskit import (QuantumCircuit, Aer, execute)

def add2(circuit):
   circuit.toffoli([0, 1, 1, 3], [2, 3, 5, 5], [5, 6, 6, 6])
   circuit.barrier()
   circuit.cx(range(4),[4,5,4,5])

circuit = QuantumCircuit(7, 3)
circuit.x(0)
circuit.x(3)

circuit.barrier()

add2(circuit)

circuit.barrier()

circuit.measure(range(4, 7), range(3))

print("Circuit:\n%s" % circuit)

s = Aer.get_backend('qasm_simulator')
job = execute(circuit, s, shots=1000)
result = job.result()

print("Counts:", result.get_counts())

print("Now reverse:")


circuit = QuantumCircuit(7, 7)

circuit.x(4)
circuit.x(5)
circuit.h(range(7))

circuit.barrier()

cc = QuantumCircuit(7,7)
add2(cc)
cc.inverse()

circuit += cc

circuit.barrier()

circuit.h(range(4))

circuit.barrier()

circuit.measure(range(4), range(4))

print(circuit)

job = execute(circuit, s, shots=1000)
print("Got this:", job.result().get_counts())
