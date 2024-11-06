# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed234.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi
from random import randrange
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(1)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",1,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(2)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",2,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(3)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",3,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(4)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",4,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(5)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",5,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(6)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",6,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(7)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",7,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(8)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",8,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(9)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",9,", the best k is",best_k)
print()
# we try each angle of the form k*2*pi/11 for k=1,...,10
# we try to find the best k for which we observe 1 the most
number_of_one_state = 0
best_k = 1
all_outcomes_for_i = "length "+str(10)+"-> "
theta = 1*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(1)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 1
theta = 2*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(2)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 2
theta = 3*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(3)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 3
theta = 4*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(4)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 4
theta = 5*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(5)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 5
theta = 6*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(6)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 6
theta = 7*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(7)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 7
theta = 8*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(8)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 8
theta = 9*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(9)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 9
theta = 10*2*pi/11
# quantum circuit with one qubit and one bit
qreg =  QuantumRegister(1) 
creg = ClassicalRegister(1) 
mycircuit = QuantumCircuit(qreg,creg)
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.ry(2*theta,qreg[0]) # apply one rotation for each symbol
mycircuit.measure(qreg[0],creg[0])
# execute the circuit 10000 times
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=10000)
counts = job.result().get_counts(mycircuit)
all_outcomes_for_i = all_outcomes_for_i + str(10)+ ":" + str(counts['1']) + "  "
if int(counts['1']) > number_of_one_state:
    number_of_one_state = counts['1']
    best_k = 10
print(all_outcomes_for_i)
print("for length",10,", the best k is",best_k)
print()
