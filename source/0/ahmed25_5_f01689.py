# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed25_5.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi

# initialize the circuit
qreg = QuantumRegister(4)  
circuit = QuantumCircuit(qreg)

# we use the fourth qubit as the auxiliary

# apply a rotation to the first qubit when the third and second qubits are in states |0> and |1>
# change the state of the third qubit to |1>
circuit.x(qreg[2])
# if both the third and second qubits are in states |1>, the state of auxiliary qubit is changed to |1> 
circuit.ccx(qreg[2],qreg[1],qreg[3])
# the rotation is applied to the first qubit if the state of auxiliary qubit is |1>
circuit.cu3(2*pi/6,0,0,qreg[3],qreg[0])
# reverse the effects
circuit.ccx(qreg[2],qreg[1],qreg[3])
circuit.x(qreg[2])

circuit.draw()

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi,sin

# the angles of rotations
theta1 = pi/10
theta2 = 2*pi/10
theta3 = 3*pi/10
theta4 = 4*pi/10

# for verification, print sin(theta)'s
print("sin(theta1) = ",round(sin(theta1),3))
print("sin(theta2) = ",round(sin(theta2),3))
print("sin(theta3) = ",round(sin(theta3),3))
print("sin(theta4) = ",round(sin(theta4),3))
print()

qreg =  QuantumRegister(4) 
circuit = QuantumCircuit(qreg)

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

# read the corresponding unitary matrix
job = execute(circuit,Aer.get_backend('unitary_simulator'),optimization_level=0)
unitary_matrix=job.result().get_unitary(circuit,decimals=3)
for i in range(len(unitary_matrix)):
    s=""
    for j in range(len(unitary_matrix)):
        val = str(unitary_matrix[i][j].real)
        while(len(val)<7): val  = " "+val
        s = s + val
    print(s)
