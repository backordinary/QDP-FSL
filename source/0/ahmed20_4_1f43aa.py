# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed20_4.py
# import all necessary objects and methods for quantum circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

all_inputs=['00','01','10','11']

for input in all_inputs:
    qreg2 =  QuantumRegister(2) # quantum register with 2 qubits
    creg2 = ClassicalRegister(2) # classical register with 2 bits
    mycircuit2 = QuantumCircuit(qreg2,creg2) # quantum circuit with quantum and classical registers
    
    #initialize the inputs
    if input[0]=='1':
        mycircuit2.x(qreg2[0]) # set the state of the first qubit to |1>
    if input[1]=='1':
        mycircuit2.x(qreg2[1]) # set the state of the second qubit to |1>

    # apply h-gate to both qubits
    mycircuit2.h(qreg2[0])
    mycircuit2.h(qreg2[1])

    # apply cx(first-qubit,second-qubit)
    mycircuit2.cx(qreg2[0],qreg2[1])

    # apply h-gate to both qubits
    mycircuit2.h(qreg2[0])
    mycircuit2.h(qreg2[1])

    # measure both qubits
    mycircuit2.measure(qreg2,creg2)
    
    # execute the circuit 100 times in the local simulator
    job = execute(mycircuit2,Aer.get_backend('qasm_simulator'),shots=100)
    counts = job.result().get_counts(mycircuit2)
    for outcome in counts: # print the reverse of the outcomes
        reverse_outcome = ''
        for i in outcome:
            reverse_outcome = i + reverse_outcome
        print("our input is",input,": ",reverse_outcome,"is observed",counts[outcome],"times")
        
        
