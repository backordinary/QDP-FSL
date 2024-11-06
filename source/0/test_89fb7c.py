# https://github.com/gerhard-st/qk-test/blob/568ef4231c1eeedfeb116b801ce22636865b5980/test.py

from qiskit import *
from math import pi
#from qiskit.visualization import plot_bloch_multivector

backend = Aer.get_backend('statevector_simulator')

print("_________________________________________")
#
qc = QuantumCircuit(1,1)
out = execute(qc,backend).result().get_statevector()
print(qc)
print(out)
print("_________________________________________")

#
qc.x(0)
out = execute(qc,backend).result().get_statevector()
print(qc)
print(out)
print("_________________________________________")

#
qc.h(0)
out = execute(qc,backend).result().get_statevector()
print(qc)
print(out)
print("_________________________________________")

#
qc.barrier()
qc.measure(0,0)
print(qc)
print(execute(qc,backend).result().get_statevector())
print("_________________________________________")

job = backend.run(qc, shots=1, memory=True)
output  = job.result().get_memory()[0]
#print(job.result())
print("\nMeasurement is %s" %output)
print("_________________________________________")
