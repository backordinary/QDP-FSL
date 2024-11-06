# https://github.com/pricoptudor/Quantum/blob/acf20bcc97cc22e3c4584f876954c434092b3360/Homework-1/ex3.py
from qiskit import *
from qiskit.visualization import *
import numpy as np
import cmath

#circuitul U care cloneaza |+>,|->:
circuit=QuantumCircuit(2,2)

circuit.h(0)
circuit.i(1)

circuit.cx(0,1)

circuit.h(0)
circuit.h(1)

circuit2=QuantumCircuit(2,2)

#circuit2.h(0)
#circuit2.barrier()
circuit=circuit2.compose(circuit)

print("Poarta de clonare este:")
print(circuit.draw())

#testare |00>:
test=QuantumCircuit(2,2)

test=test.compose(circuit)
test.barrier()
test.measure(0,0)
test.measure(1,1)

print(test.draw())

simulator=Aer.get_backend('qasm_simulator')
result=execute(test,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/ex3.1.png')

#testare |10>:
test=QuantumCircuit(2,2)

test.x(0)
test.barrier()
test=test.compose(circuit)
test.barrier()
test.measure(0,0)
test.measure(1,1)

print(test.draw())

result=execute(test,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/ex3.2.png')

print('Pentru inputul |00> si |10> starea din primul qubit nu se copiaza')
print('In schimb se ajunge la una din starile lui Bell')