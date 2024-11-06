# https://github.com/HindeDS/Gates-Questions/blob/29ea219923c743c92a943790deb8a47d8de0cd75/Gates-Questions.py
# import random 
from qiskit import*
#choose the number of qubits
m = int(input("Enter the number of qubits:- "))
#create the circuit
qc=QuantumCircuit(m)
#Apply X gate to last qubit
qc.x(m-1)
backend1 = Aer.get_backend("statevector_simulator")
result1 = execute(qc, backend=backend1, shots=1).result()
counts1 = result1.get_counts(qc)
#First question
p1 = input("what is the outcome after applying x gate to last qubit?:- ")
#First answer
for outcome1 in counts1:
    if (p1==outcome1):
        print("correct answer")
    else:
        print("uncorrect answer,the correct answer is :",outcome1)
#Apply X gate to first qubit   
qc.x(m-m)
backend2 = Aer.get_backend("statevector_simulator")
result2 = execute(qc, backend=backend2, shots=1).result()
counts2 = result2.get_counts(qc)
#Second question
p2 = input("what is the outcome after applying x gate to first qubit qubit?:- ")
#Second Answer
for outcome2 in counts2:
    if (p2==outcome2):
        print("correct answer")
    else:
        print("uncorrect answer,the correct answer is :",outcome2)
#Apply Y gate to last qubit
qc.y(m-1)
backend3 = Aer.get_backend("statevector_simulator")
result3 = execute(qc, backend=backend3, shots=1).result()
counts3 = result3.get_counts(qc)
#Third Question
p3 = input("what is the outcome after applying Y gate to last qubit qubit?:- ")
#Third Answer
for outcome3 in counts3:
    if (p3==outcome3):
        print("correct answer")
    else:
        print("uncorrect answer,the correct answer is :",outcome3)
#Apply Y gate to first qubit
qc.y(m-m)
backend4 = Aer.get_backend("statevector_simulator")
result4 = execute(qc, backend=backend4, shots=1).result()
counts4 = result4.get_counts(qc)
#Foorth Question
p4 = input("what is the outcome after applying Y gate to first qubit?:- ")
#Foorth Answer
for outcome4 in counts4:
    if (p4==outcome4):
        print("correct answer")
    else:
        print("uncorrect answer,the correct answer is :",outcome4)
#Apply H gate to first gate
qc.h(m-m)
backend5 = Aer.get_backend("statevector_simulator")
result5 = execute(qc, backend=backend5, shots=1).result()
counts5 = result5.get_counts(qc)
#Fith Question
p5 = input("what is the first state after applying h gate to first qubit?:- ")
#sixth Question
p6 = input("what is the second state after applying h gate to first qubit?:- ")
#Answers for fifth and sixth questions
for outcome5 in counts5:
    if (p5==outcome5):
        print("correct answer for the first state")
        
    elif (p6==outcome5):
        print("correct answer for the second state")
    else:
        print("uncorrect answer,the correct answer is :",outcome5)
