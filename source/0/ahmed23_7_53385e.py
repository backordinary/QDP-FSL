# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed23_7.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi
from random import randrange

number_of_qubits = 4
#number_of_qubits = 5
# randomly picked angles of rotations 
theta = []
for i in range(number_of_qubits):
    k =  randrange(1,31)
    print("k",str(i),"=",k)
    theta += [k*2*pi/31]
# print(theta)

# we count the number of zeros
zeros = ''
for i in range(number_of_qubits):
    zeros = zeros + '0'
print("zeros = ",zeros)
print()

max_percentange = 0
# we read streams of length from 1 to 30
for i in range(1,31):
    # quantum circuit with qubits and bits
    qreg =  QuantumRegister(number_of_qubits) 
    creg = ClassicalRegister(number_of_qubits) 
    mycircuit = QuantumCircuit(qreg,creg)
    # the stream of length i
    for j in range(i):
        # apply rotations for each symbol
        for k in range(number_of_qubits):
            mycircuit.ry(2*theta[k],qreg[k]) 
    # we measure after reading the whole stream
    mycircuit.measure(qreg,creg)
    # execute the circuit N times
    N = 1000
    job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=N)
    counts = job.result().get_counts(mycircuit)
    # print(counts)
    if zeros in counts.keys():
        c = counts[zeros]
    else:
        c = 0
    # print('000 is observed',c,'times out of',N)
    percentange = round(c/N*100,1)
    if max_percentange < percentange: max_percentange = percentange
    # print("the ration of 000 is ",percentange,"%")
    # print()
print("max percentage is",max_percentange)
