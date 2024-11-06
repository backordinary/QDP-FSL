# https://github.com/pricoptudor/Quantum/blob/acf20bcc97cc22e3c4584f876954c434092b3360/Homework-1/ex1.py
from qiskit import *
from qiskit.visualization import *

n=3

circuit=QuantumCircuit(n,n)

circuit.x(0)
circuit.barrier()
circuit.h(1)
circuit.cx(1,2)
circuit.barrier()

#'circuit' : din |000> in starea |x>;
'''verificare pentru starea |x>:
for i in range(n):
    circuit.measure(i,i)

simulator=Aer.get_backend('qasm_simulator')
result=execute(circuit,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/ex1.png')

print(circuit.draw())'''

#circuit pentru transformare |x> in |x{tilda}>:

circuit.x(0)
circuit.i(1)
circuit.i(2)

circuit.barrier()

circuit.i(0)
circuit.x(1)
circuit.i(2)

circuit.barrier()

for q in range(n):
    circuit.measure(q,q)

print(circuit.draw())

simulator=Aer.get_backend('qasm_simulator')
result=execute(circuit,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/ex1.png')

