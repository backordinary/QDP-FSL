# https://github.com/Sassa-nf/pi/blob/57fabb0c4957a1133790a30410b7b68348e0e2f2/quant/toffoli.py
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)

#simulator = Aer.get_backend('qasm_simulator')
simulator = Aer.get_backend('unitary_simulator')
circuit = QuantumCircuit(2, 2)

circuit.t(1)
circuit.cx(0, 1)
circuit.tdg(1)
circuit.t(0)
circuit.cx(0,1)

print("\nCircuit:\n%s" % circuit)

#circuit.measure([0,1], [0,1])

#job = execute(circuit, simulator, shots=10)
job = execute(circuit, simulator)
result = job.result()
#counts = result.get_counts(circuit)
#print("\nTotal count for 00 and 11 are:",counts)
print("\n%s" % result.get_unitary(circuit, decimals=3)) # this is interesting: q_0 is modified "only" by T, but remains unchanged, if q_1 is 0
