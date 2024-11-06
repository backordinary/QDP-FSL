# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed20_5.py
# import all necessary objects and methods for quantum circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

all_inputs=['00','01','10','11']

for input in all_inputs:
    qreg3 =  QuantumRegister(2) # quantum register with 2 qubits
    creg3 = ClassicalRegister(2) # classical register with 2 bits
    mycircuit3 = QuantumCircuit(qreg3,creg3) # quantum circuit with quantum and classical registers
    
    #initialize the inputs
    if input[0]=='1':
        mycircuit3.x(qreg3[0]) # set the value of the first qubit to |1>
    if input[1]=='1':
        mycircuit3.x(qreg3[1]) # set the value of the second qubit to |1>

    # apply cx(first-qubit,second-qubit)
    mycircuit3.cx(qreg3[0],qreg3[1])
    # apply cx(second-qubit,first-qubit)
    mycircuit3.cx(qreg3[1],qreg3[0])
    # apply cx(first-qubit,second-qubit)
    mycircuit3.cx(qreg3[0],qreg3[1])
    
    mycircuit3.measure(qreg3,creg3)
    
    # execute the circuit 100 times in the local simulator
    job = execute(mycircuit3,Aer.get_backend('qasm_simulator'),shots=100)
    counts = job.result().get_counts(mycircuit3)
    for outcome in counts: # print the reverse of the outcomes
        reverse_outcome = ''
        for i in outcome:
            reverse_outcome = i + reverse_outcome
        print("our input is",input,": ",reverse_outcome,"is observed",counts[outcome],"times")
