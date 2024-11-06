# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed25_3.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi, sin, cos

# the angle of rotation
theta1 = pi/23
theta2 = 2*pi/23
theta3 = 4*pi/23


precision = 3

print("a1 = theta3 => sin(a1) = ",round(sin(theta3),precision))
print("a2 = theta2+theta3 => sin(a2) = ",round(sin(theta2+theta3),precision))
print("a3 = theta1 => sin(a3) = ",round(sin(theta1),precision))
print("a4 = theta1+theta2 => sin(a4) = ",round(sin(theta1+theta2),precision))
print()

qreg = QuantumRegister(3) 
creg = ClassicalRegister(3) 
circuit = QuantumCircuit(qreg,creg)

# controlled rotation when the third qubit is |1>
circuit.cu3(2*theta1,0,0,qreg[2],qreg[0])

# controlled rotation when the second qubit is |1>
circuit.cu3(2*theta2,0,0,qreg[1],qreg[0])

# controlled rotation when the third qubit is |0>
circuit.x(qreg[2])
circuit.cu3(2*theta3,0,0,qreg[2],qreg[0])
circuit.x(qreg[2])

# read the corresponding unitary matrix
job = execute(circuit,Aer.get_backend('unitary_simulator'),optimization_level=0)
unitary_matrix=job.result().get_unitary(circuit,decimals=precision)
for i in range(len(unitary_matrix)):
    s=""
    for j in range(len(unitary_matrix)):
        val = str(unitary_matrix[i][j].real)
        while(len(val)<precision+4): val  = " "+val
        s = s + val
    print(s)
