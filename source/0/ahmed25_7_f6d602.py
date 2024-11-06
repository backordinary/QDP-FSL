# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed25_7.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi,sin
from random import randrange

# the angle of rotation
k1 = randrange(1,61)
theta1 = k1*2*pi/61
k2 = randrange(1,61)
theta2 = k2*2*pi/61
k3 = randrange(1,61)
theta3 = k3*2*pi/61
k4 = randrange(1,61)
theta4 = k4*2*pi/61

max_percentange = 0

# for each stream of length of 1, 11, 21, 31, 41, 51, and 61
for i in [1,11,21,31,41,51,61]: 
#for i in range(1,62): 
    # initialize the circuit
    qreg =  QuantumRegister(4) 
    creg = ClassicalRegister(4)
    circuit = QuantumCircuit(qreg,creg)

    # Hadamard operators before reading the stream
    for m in range(3):
        circuit.h(qreg[m])   
        
    # read the stream of length i
    print("stream of length",i,"is being read")
    for j in range(i):         
        # the third qubit is in |0>
        # the second qubit is in |0>
        circuit.x(qreg[2])
        circuit.x(qreg[1])
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.cu3(2*theta1,0,0,qreg[3],qreg[0])
        # reverse the effects
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.x(qreg[1])
        circuit.x(qreg[2])


        # the third qubit is in |0>
        # the second qubit is in |1>
        circuit.x(qreg[2])
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.cu3(2*theta2,0,0,qreg[3],qreg[0])
        # reverse the effects
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.x(qreg[2])

        # the third qubit is in |1>
        # the second qubit is in |0>
        circuit.x(qreg[1])
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.cu3(2*theta3,0,0,qreg[3],qreg[0])
        # reverse the effects
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.x(qreg[1])

        # the third qubit is in |1>
        # the second qubit is in |1>
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        circuit.cu3(2*theta4,0,0,qreg[3],qreg[0])
        # reverse the effects
        circuit.ccx(qreg[2],qreg[1],qreg[3])
        
    # Hadamard operators after reading the stream
    for m in range(3):
        circuit.h(qreg[m])  
    # we measure after reading the whole stream
    circuit.measure(qreg,creg)
    # execute the circuit N times
    N = 1000
    job = execute(circuit,Aer.get_backend('qasm_simulator'),shots=N)
    counts = job.result().get_counts(circuit)
    print(counts)
    if '0000' in counts.keys():
        c = counts['0000']
    else:
        c = 0
    print('0000 is observed',c,'times out of',N)
    percentange = round(c/N*100,1)
    if max_percentange < percentange and i != 61: max_percentange = percentange
    print("the ration of 0000 is ",percentange,"%")
    print()  
print("maximum percentage of observing unwanted '0000' is",max_percentange)
