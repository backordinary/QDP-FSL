# https://github.com/huub-d96/QA_Stage/blob/357df9de7c16f21bb6deae1ede4ebe0bbf761788/teleport.py
"""
Created on Thu Aug  5 12:05:10 2021

Auhtor: Huub Donkers
Project: Exploring XACC Framework
Description: Test script for quantum teleportation 

"""


import xacc

#Setup qpu platform for two qubits
qpu = xacc.getAccelerator('ibm:ibmq_lima', {'shots':2048})
buffer = xacc.qalloc(3)

#Create quantum program
compiler = xacc.getCompiler('xasm')
circuit = '''
        __qpu__ void teleport(qbit q){
        
            //Initialize teleported qubit to 1
            X(q[0]);
            
            //Create entangled pair between Alice and Bob
    		H(q[1]);
    		CX(q[1],q[2]);
            
            //Perform measurements (Alice)
            CX(q[0],q[1]);
            H(q[0]);
    		Measure(q[0], c[0]);
    		Measure(q[1], c[1]);
            
            //Corrections (Bob)            
            CX(q[1], q[2]);
            CZ(q[0], q[2]);
            
            // Measure teleported qubit
            Measure(q[2]);
		}
        '''

program = compiler.compile(circuit, qpu)

mapped_program = program.getComposite('teleport')
mapped_program.defaultPlacement(qpu)

#transform = xacc.getIRTransformation('circuit-optimizer')
#transform.apply(mapped_program, qpu, {})

print('HOWDY QASM:\n', qpu.getNativeCode(mapped_program, {'format': 'qasm'}))
# we can also see the native circuit as a QObj json
print('HOWDY QObj:\n', qpu.getNativeCode(mapped_program, {'format': 'QObj'}))

from qiskit import QuantumCircuit
qiskit_qc = QuantumCircuit.from_qasm_str(qpu.getNativeCode(mapped_program, {'format': 'qasm'}))
print(qiskit_qc)
qiskit_qc.draw()


#Execute and readout buffer
qpu.execute(buffer, mapped_program)
results = buffer.getMeasurementCounts()
print(results)
