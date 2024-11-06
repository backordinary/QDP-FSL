# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed28_3.py
def inversion(circuit,quantum_reg):
    #step 1
    circuit.x(quantum_reg[2])
    circuit.h(quantum_reg[2])
    
    #step 2
    circuit.h(quantum_reg[1])
    circuit.h(quantum_reg[0])
    circuit.x(quantum_reg[1])
    circuit.x(quantum_reg[0])

    #step 3
    circuit.ccx(quantum_reg[1],quantum_reg[0],quantum_reg[2])
    
    #step 4
    circuit.x(quantum_reg[2])

    #step 5
    circuit.x(quantum_reg[1])
    circuit.x(quantum_reg[0])
    circuit.h(quantum_reg[1])
    circuit.h(quantum_reg[0])

    #step 6
    circuit.h(quantum_reg[2])
    circuit.x(quantum_reg[2])
    
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

qreg4 =  QuantumRegister(3)
creg4 = ClassicalRegister(3)

mycircuit4 = QuantumCircuit(qreg4,creg4)

inversion(mycircuit4,qreg4)

job = execute(mycircuit4,Aer.get_backend('unitary_simulator'))
u=job.result().get_unitary(mycircuit4,decimals=3)
for i in range(len(u)):
    s=""
    for j in range(len(u)):
        val = str(u[i][j].real)
        while(len(val)<5): val  = " "+val
        s = s + val
    print(s)
    
mycircuit4.draw(output='mpl')
