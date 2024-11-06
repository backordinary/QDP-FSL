# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed24_2.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

qreg5 =  QuantumRegister(9)
creg5 = ClassicalRegister(9)

mycircuit5 = QuantumCircuit(qreg5,creg5)

mycircuit5.x(qreg5[7])
mycircuit5.x(qreg5[5])

mycircuit5.ccx(qreg5[8],qreg5[7],qreg5[3])
mycircuit5.ccx(qreg5[6],qreg5[5],qreg5[2])
mycircuit5.ccx(qreg5[4],qreg5[3],qreg5[1])

mycircuit5.ccx(qreg5[2],qreg5[1],qreg5[0])

# Returning additional and control qubits to the initial state
mycircuit5.ccx(qreg5[4],qreg5[3],qreg5[1])
mycircuit5.ccx(qreg5[6],qreg5[5],qreg5[2])
mycircuit5.ccx(qreg5[8],qreg5[7],qreg5[3])
mycircuit5.x(qreg5[5])
mycircuit5.x(qreg5[7])

mycircuit5.measure(qreg5,creg5)

job = execute(mycircuit5,Aer.get_backend('qasm_simulator'),shots=10000)
counts5 = job.result().get_counts(mycircuit5)
print(counts5)

mycircuit5.draw(output="mpl")
