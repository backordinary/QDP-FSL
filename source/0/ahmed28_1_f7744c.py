# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed28_1.py
#number - marked element, between 0 and 3.
def query(circuit,quantum_reg,number):
    # prepare ancilla qubit
    circuit.x(quantum_reg[2])
    circuit.h(quantum_reg[2])

    if(number%2 == 0):
        circuit.x(quantum_reg[0])
    if(number < 2):
        circuit.x(quantum_reg[1])
    circuit.ccx(quantum_reg[0],quantum_reg[1],quantum_reg[2])
    if(number < 2):
        circuit.x(quantum_reg[1])
    if(number%2 == 0):
        circuit.x(quantum_reg[0])

    # put ancilla qubit back into state |0>
    circuit.h(quantum_reg[2])
    circuit.x(quantum_reg[2])
    
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

qreg3 =  QuantumRegister(3)
creg3 = ClassicalRegister(3)

mycircuit3 = QuantumCircuit(qreg3,creg3)

#Any value between 0 and 3.
query(mycircuit3,qreg3,1)
#Uncomment the next line to mark additional element.
#query(mycircuit3,qreg3,2)

job = execute(mycircuit3,Aer.get_backend('unitary_simulator'))
u=job.result().get_unitary(mycircuit3,decimals=3)
for i in range(len(u)):
    s=""
    for j in range(len(u)):
        val = str(u[i][j].real)
        while(len(val)<5): val  = " "+val
        s = s + val
    print(s)
