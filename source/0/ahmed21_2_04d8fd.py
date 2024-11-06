# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed21_2.py
# import all necessary objects and methods for quantum circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

# Create a circuit with 7 qubits.
n = 7
qreg2 =  QuantumRegister(n) # quantum register with 7 qubits
creg2 = ClassicalRegister(n) # classical register with 7 bits

mycircuit2 = QuantumCircuit(qreg2,creg2) # quantum circuit with quantum and classical registers

# the first six qubits are already in |0>

# set the last qubit to |1>
mycircuit2.x(qreg2[n-1]) # apply x-gate (NOT operator)

# apply Hadamard to all qubits.
for i in range(n):
    mycircuit2.h(qreg2[i])


# apply CNOT operator (first-qubit,last-qubit) 
# apply CNOT operator (fourth-qubit,last-qubit) 
# apply CNOT operator (fifth-qubit,last-qubit)
mycircuit2.cx(qreg2[0],qreg2[n-1])
mycircuit2.cx(qreg2[3],qreg2[n-1])
mycircuit2.cx(qreg2[4],qreg2[n-1])

# apply Hadamard to all qubits.
for i in range(n):
    mycircuit2.h(qreg2[i])

# define a barrier
mycircuit2.barrier()

# measure all qubits
mycircuit2.measure(qreg2,creg2)

# execute the circuit 100 times in the local simulator

job = execute(mycircuit2,Aer.get_backend('qasm_simulator'),shots=100)
counts = job.result().get_counts(mycircuit2)

# print the reverse of the outcome
for outcome in counts:
    reverse_outcome = ''
    for i in outcome:
        reverse_outcome = i + reverse_outcome
    print(reverse_outcome,"is observed",counts[outcome],"times")
    for i in range(len(reverse_outcome)):
        print("the final value of the qubit nr.",(i+1),"is",reverse_outcome[i])
