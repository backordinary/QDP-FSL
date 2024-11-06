# https://github.com/pricoptudor/Quantum/blob/acf20bcc97cc22e3c4584f876954c434092b3360/Homework-1/ex2.py
from qiskit import *
from qiskit.visualization import *
import numpy as np
import cmath

'''
#subpunctul i):

circuit=QuantumCircuit(2,2)

circuit.h(1)
circuit.cx(0,1)
circuit.h(1)

circuit.barrier()

for q in range(2):
    circuit.measure(q,q)

print(circuit.draw())

simulator=Aer.get_backend('qasm_simulator')
result=execute(circuit,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/exe2i.png')

#subpunctul ii):
circuit2=QuantumCircuit(2,2)

circuit2.h(0)
circuit2.barrier()

circuit2=circuit2.compose(circuit)

print(circuit2.draw())

result=execute(circuit2,simulator,shots=1024).result()
counts=result.get_counts()
print(counts)
plot_histogram(counts).savefig('homework/exe2ii.png')'''

#subpunctul iii):

#testare rezultate pe fiecare input posibil:
def rulare(circuit_input):
    firstCircuit=QuantumCircuit(2,2)

    firstCircuit.h(1)
    firstCircuit.cx(0,1)
    firstCircuit.h(1)

    firstCircuit=circuit_input.compose(firstCircuit)

    firstCircuit.measure(0,0)
    firstCircuit.measure(1,1)

    print(firstCircuit.draw())

    simulator=Aer.get_backend('qasm_simulator')
    result=execute(firstCircuit,simulator,shots=1024).result()
    counts =result.get_counts()
    print(counts)


    secondCircuit=QuantumCircuit(2,2)

    secondCircuit.cz(1,0)

    secondCircuit=circuit_input.compose(secondCircuit)

    secondCircuit.measure(0,0)
    secondCircuit.measure(1,1)
    
    print(secondCircuit.draw())

    result=execute(secondCircuit,simulator,shots=1024).result()
    counts=result.get_counts()
    print(counts)
    print("")

#input|00>:
circuit_input=QuantumCircuit(2,2)

circuit_input.barrier()

print("Inputul |00>:")
rulare(circuit_input)

#input|01>:
circuit_input=QuantumCircuit(2,2)

circuit_input.x(1)
circuit_input.barrier()

print("Inputul |01>:")
rulare(circuit_input)

#input|10>:
circuit_input=QuantumCircuit(2,2)

circuit_input.x(0)
circuit_input.barrier()

print("Inputul |10>:")
rulare(circuit_input)

#input|11>:
circuit_input=QuantumCircuit(2,2)

circuit_input.x(0)
circuit_input.x(1)
circuit_input.barrier()

print("Inputul |11>:")
rulare(circuit_input)



print("////////////////////////")
print("")
#matrici circuite:
firstCircuit=QuantumCircuit(2,2)

firstCircuit.h(1)
firstCircuit.cx(0,1)
firstCircuit.h(1)

print(firstCircuit.draw())

simulator=Aer.get_backend('unitary_simulator')
result=execute(firstCircuit,simulator).result()
matrice1=result.get_unitary(firstCircuit)
print("Matricea primului circuit este:\n",matrice1)
print("")


secondCircuit=QuantumCircuit(2,2)

secondCircuit.cz(1,0)
    
print(secondCircuit.draw())

result=execute(secondCircuit,simulator).result()
matrice2=result.get_unitary(secondCircuit)
print("Matricea celui de-al doilea circuit este:\n",matrice2)
print("")


def egalitate_matrici(m1,m2):
    if(len(m1)!=len(m2)):
        return ('Dimensiuni incompatibile')
    if(len(m1[0])!=len(m2[0])):
        return ('Dimensiuni incompatibile')
    for i in range(len(m1)):
        for j in range(len(m2)):
            if(abs(m1[i][j].real-m2[i][j].real)>np.finfo(float).eps):
                return ('Not equal')
            if(abs(m1[i][j].imag-m2[i][j].imag)>np.finfo(float).eps):
                return ('Not equal')
    return ('Matrici egale')

print(egalitate_matrici(matrice1,matrice2))
print(np.finfo(float).eps)

