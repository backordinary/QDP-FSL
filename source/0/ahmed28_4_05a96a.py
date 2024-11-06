# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed28_4.py
def big_inversion(circuit,quantum_reg):
    circuit.x(quantum_reg[4])
    circuit.h(quantum_reg[4])
    circuit.h(quantum_reg[0])
    circuit.x(quantum_reg[0])
    circuit.h(quantum_reg[1])
    circuit.x(quantum_reg[1])
    circuit.h(quantum_reg[2])
    circuit.x(quantum_reg[2])

    circuit.ccx(quantum_reg[1],quantum_reg[0],quantum_reg[3])
    circuit.ccx(quantum_reg[2],quantum_reg[3],quantum_reg[4])
    circuit.ccx(quantum_reg[1],quantum_reg[0],quantum_reg[3])
    
    circuit.x(quantum_reg[4])
    circuit.x(quantum_reg[0])
    circuit.h(quantum_reg[0])
    circuit.x(quantum_reg[1])
    circuit.h(quantum_reg[1])
    circuit.x(quantum_reg[2])
    circuit.h(quantum_reg[2])

    circuit.h(quantum_reg[4])
    circuit.x(quantum_reg[4])
    
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

big_qreg2 =  QuantumRegister(5)
big_creg2 = ClassicalRegister(5)

big_mycircuit2 = QuantumCircuit(big_qreg2,big_creg2)

big_inversion(big_mycircuit2,big_qreg2)

job = execute(big_mycircuit2,Aer.get_backend('unitary_simulator'))
u=job.result().get_unitary(big_mycircuit2,decimals=3)
s=""
val = str(u[0][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[0][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[1][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[1][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[2][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[2][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[3][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[3][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[4][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[4][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[5][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[5][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[6][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[6][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
s=""
val = str(u[7][0].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][1].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][2].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][3].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][4].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][5].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][6].real)
while(len(val)<6): val  = " "+val
s = s + val
val = str(u[7][7].real)
while(len(val)<6): val  = " "+val
s = s + val
print(s)
