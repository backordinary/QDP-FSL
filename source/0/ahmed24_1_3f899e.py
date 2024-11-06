# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed24_1.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

qreg4 =  QuantumRegister(7)
creg4 = ClassicalRegister(7)

mycircuit4 = QuantumCircuit(qreg4,creg4)

mycircuit4.x(qreg4[6])
mycircuit4.x(qreg4[5])
mycircuit4.x(qreg4[4])
mycircuit4.x(qreg4[3])

mycircuit4.ccx(qreg4[6],qreg4[5],qreg4[2])
mycircuit4.ccx(qreg4[4],qreg4[3],qreg4[1])
mycircuit4.ccx(qreg4[2],qreg4[1],qreg4[0])

# Returning additional qubits to the initial state
mycircuit4.ccx(qreg4[4],qreg4[3],qreg4[1])
mycircuit4.ccx(qreg4[6],qreg4[5],qreg4[2])

mycircuit4.measure(qreg4,creg4)

job = execute(mycircuit4,Aer.get_backend('qasm_simulator'),shots=10000)
counts4 = job.result().get_counts(mycircuit4)
print(counts4)

mycircuit4.draw(output="mpl")
