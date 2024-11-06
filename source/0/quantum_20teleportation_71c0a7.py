# https://github.com/NavMohan-24/Quantum-Communication/blob/f989d78de6e18503597898f66a4de90d3cc16cd1/Quantum%20Teleportation.py
#!/usr/NavMohan-24
# coding: utf-8

from qiskit import*
from qiskit.tools.visualization import*
from qiskit.quantum_info import random_statevector
from qiskit.extensions import Initialize

qr=QuantumRegister(3,'qr')
crx=ClassicalRegister(1,'crx')
crz=ClassicalRegister(1,'crz')
qc=QuantumCircuit(qr,crx,crz)
#qc.draw('mpl')


#creation of entanglement channel for communication (1/sqrt(2)(|00>+|11>))
def create_ent_channel(qc,a,b):
    qc.barrier()
    qc.h(qr[a])
    qc.cx(qr[a],qr[b])
    
#Bell Measurement by alice
def bell_measurement(qc,a,b):
    qc.barrier()
    qc.cx(qr[a],qr[b])
    qc.h(qr[a])
    qc.measure(qr[a],crx)
    qc.measure(qr[b],crz)
    

#Bobs transformation
def bob_transformation(qc,qubit):
    qc.barrier()
    qc.x(qubit).c_if(crx,1)
    qc.z(qubit).c_if(crz,1)
#note: bob's transformation may change depending upon the entanglement channel

#creating a state that we want to teleport : alice will be the sender
state=random_statevector(2,seed=3)
state.probabilities()
init_gate=Initialize(state.data)
qc.append(init_gate,[0])



create_ent_channel(qc,1,2)
bell_measurement(qc,0,1)
bob_transformation(qc,2)
#qc.draw('mpl')

#thus the bob qubit will be transformed to state which alice want to sent. 
#state of the alice's qubit get collapsed during the process

'''#checking the bobs state
backend=BasicAer.get_backend('statevector_simulator')
in_state=state
plot_bloch_multivector(in_state)'''



disentangler=init_gate.gates_to_uncompute() #re
qc.append(disentangler,[2]) 

'''out_state=execute(qc,backend).result().get_statevector()
plot_bloch_multivector(out_state)'''


#to check bob received corect state
qc.measure(qr[2],crx)
#qc.draw('mpl')


backend=BasicAer.get_backend('qasm_simulator')
counts=execute(qc,backend,shots=1024).result().get_counts()
plot_histogram(counts)

qc.draw('mpl')
