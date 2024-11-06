# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed27_1.py
# import all necessary objects and methods for quantum circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

qreg = QuantumRegister(4) # quantum register with 4 qubits
creg = ClassicalRegister(4) # classical register with 4 bits
mycircuit = QuantumCircuit(qreg,creg) # quantum circuit with quantum and classical registers
mycircuit.h(qreg[0])
mycircuit.h(qreg[1])
mycircuit.h(qreg[2])
mycircuit.h(qreg[3])

# measure all qubits
mycircuit.measure(qreg,creg)
    
# execute the circuit 1600 times, and print the outcomes
job = execute(mycircuit,Aer.get_backend('qasm_simulator'),shots=1600)
counts = job.result().get_counts(mycircuit)
for outcome in counts:
    reverse_outcome = ''
    for i in outcome:
        reverse_outcome = i + reverse_outcome
    print(reverse_outcome,"is observed",counts[outcome],"times")
