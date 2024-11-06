# https://github.com/Innanov/QuantumComputing/blob/c93d793d679e2f21ef25b7c59cbd6c4c7cd9fa5b/FirstQauntumCircuit.py
#INNAN Nouhaila 
# A quantum circuit with four quantum and classical bits 


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
circuit.draw()
