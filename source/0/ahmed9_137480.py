# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed9.py
# 
# A quantum circuit is composed by quantum and classical bits in Qiskit.
#

# here are the objects that we use to create a quantum circuit in qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# we use a quantum register to keep our quantum bits.
q =  QuantumRegister(1,"qreg") # in this example we will use a single quantum bit
# the second parameter is optional

# To retrieve an information from a quantum bit, it must be measured. (More details will appear.)
#     The measurement result is stored classically.
#     Therefore, we also use a classical regiser with classical bit(s)
c = ClassicalRegister(1,"creg") # in this example we will use a single classical bit
# the second parameter is optional

# now we can define our quantum circuit
# it is composed by a quantum and a classical registers
qc = QuantumCircuit(q,c)

# we apply operators on quantum bits
# operators are called as gates
# we apply NOT operator represented as "x" in qiskit
# operator is a part of the circuit, and we should specify the quantum bit as its parameter
qc.x(q[0]) # (quantum) bits are enumerated starting from 0
# NOT operator or x-gate is applied to the first qubit of the quantum register

# measurement is defined by associating a quantum bit to a classical bit
qc.measure(q[0],c[0])
# after the measurement, the observed value of the quantum bit is stored in the classical bit

# we run our codes until now, and then draw our circuit
print("The design of the circuit is done.")

# in Qiskit, the circuit object has a method called "draw"
# the default drawing method uses ASCII art

# let's draw our circuit now 
qc.draw()

# re-execute this cell if you DO NOT see the circuit diagram

# we can draw the same cirucuit by using matplotlib
qc.draw(output='mpl')

# we use the method "execute" and the object "Aer" from qiskit library
from qiskit import execute, Aer

# we create a job object for execution of the circuit
# there are three parameters
#     1. mycircuit
#     2. beckend on which it will be executed: we will use local simulator
#     3. how many times it will be executed, by default it is 1024
job = execute(qc,Aer.get_backend('qasm_simulator'),shots=1024)

# we can get the result of the outcome as follows
counts = job.result().get_counts(qc)
print(counts) # counts is a dictionary

# we can show the result by using histogram as follows
from qiskit.visualization import plot_histogram
plot_histogram(counts)

#print qasm code of our program
print(qc.qasm())

# 
# A quantum circuit with four quantum and classical bits 
#

# import all objects and methods at once
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

# define quantum and classical registers and then quantum circuit
q2 = QuantumRegister(4,"qreg")
c2 = ClassicalRegister(4,"creg")
qc2 = QuantumCircuit(q2,c2)

# apply x-gate to the first quantum bit twice
qc2.x(q2[0])
qc2.x(q2[0])

# apply x-gate to the fourth quantum bit once
qc2.x(q2[3])

# apply x-gate to the third quantum bit three times
qc2.x(q2[2])
qc2.x(q2[2])
qc2.x(q2[2])

# apply x-gate to the second quantum bit four times
qc2.x(q2[1])
qc2.x(q2[1])
qc2.x(q2[1])
qc2.x(q2[1])

# define a barrier (for a better visualization)
qc2.barrier()

# if the sizes of quantum and classical registers are the same, we can define measurements with a single line of code
qc2.measure(q2,c2)
# then quantum bits and classical bits are associated with respect to their indices

# run the codes until now, and then draw our circuit
print("The design of the circuit is done.")

qc2.draw(output='mpl')
# re-execute this cell if the circuit diagram does not appear

# by seting parameter "reverse_bits" to "True", the order of quantum bits are reversed when drawing

qc2.draw(output='mpl',reverse_bits=True)
# re-execute this cell if the circuit diagram does not appear

