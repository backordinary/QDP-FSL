# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed23_2.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi
from random import randrange

# the angle of rotation
r = randrange(1,11)
print("the picked angle is",r,"times of 2pi/11")
print()  
theta = r*2*pi/11

# we read streams of length from 1 to 11
for i in range(1,12):
    # quantum circuit with one qubit and one bit
    qreg =  QuantumRegister(1) 
    creg = ClassicalRegister(1) 
    mycircuit = QuantumCircuit(qreg,creg)
    # the stream of length i
    for j in range(i):
        mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
    # we measure after reading the whole stream
    mycircuit.measure(qreg[0],creg[0])
    # execute the circuit 1000 times
    job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=1000)
    counts = job.result().get_counts(mycircuit)
    print("stream of lenght",i,"->",counts)
