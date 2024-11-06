# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/build/lib/qulib/TruelyRandomByte.py

#importing modules from Qiskit library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, BasicAer, execute

def TruelyRandomByte():
    #initializing quantum register
    q = QuantumRegister(8)
    #initializing classical register
    c = ClassicalRegister(8)

    #initializing quantum circuit
    circuit = QuantumCircuit(q, c)
    circuit.h(q[0])
    circuit.h(q[1])
    circuit.h(q[2])
    circuit.h(q[3])
    circuit.h(q[4])
    circuit.h(q[5])
    circuit.h(q[6])
    circuit.h(q[7])
    
    #measuring qubit values
    circuit.measure(q, c) 
    
    #creating backend for simulation
    b = BasicAer.get_backend('qasm_simulator')
    #executing circuit on backend
    j = execute(circuit, b, shots = 1)
    #calculating result
    result = j.result()

    #listing all results
    ls = result.get_counts().keys()
    #returning result in decimal form
    for i in ls:
        return int(i,2) 
    
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())
print(TruelyRandomByte())