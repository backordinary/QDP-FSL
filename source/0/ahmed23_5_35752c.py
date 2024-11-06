# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed23_5.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi
from random import randrange

# the angles of rotations
theta1 = 3*2*pi/31
theta2 = 7*2*pi/31
theta3 = 11*2*pi/31

# we read streams of length from 1 to 30
for i in range(1,31):
    # quantum circuit with three qubits and three bits
    qreg =  QuantumRegister(3) 
    creg = ClassicalRegister(3) 
    mycircuit = QuantumCircuit(qreg,creg)
    # the stream of length i
    for j in range(i):
        # apply rotations for each symbol
        mycircuit.ry(2*theta1,qreg[0]) 
        mycircuit.ry(2*theta2,qreg[1]) 
        mycircuit.ry(2*theta3,qreg[2]) 
    # we measure after reading the whole stream
    mycircuit.measure(qreg,creg)
    # execute the circuit N times
    N = 1000
    job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=N)
    counts = job.result().get_counts(mycircuit)
    print(counts)
    if '000' in counts.keys():
        c = counts['000']
    else:
        c = 0
    print('000 is observed',c,'times out of',N)
    percentange = round(c/N*100,1)
    print("the ratio of 000 is ",percentange,"%")
    print()
