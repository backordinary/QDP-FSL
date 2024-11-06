# https://github.com/AbdulahAmer/Get-Started-with-Qiskit-Seminar/blob/dfed1d09dd01ec49b64fc83cec6852a73bf74c13/Teaching!.py
'''
Qiskit Seminar!

For reference Qiskit textbook: https://qiskit.org/textbook/preface.html
Download Python 3 : https://www.python.org/

        Open up command line and type in:
         pip install qiskit
'''
# First get all the stuff we need to code!
#Python Depends on packages to do things not contained
# IN the vanilla language


#How we will get all the goodies we need from qiskit
from qiskit import *
#initialize a place register of qubits this one is called seth and has 2 qubits
qregister = QuantumRegister(2, 'seth')
#drop our qubits into a quantum circuit!
qcircuit=QuantumCircuit(qregister)
# using .qregs tells us about quantum registers it
qcircuit.qregs


#Important python counts starting at 1
#apply H to first Qubit
qcircuit.h(0)
#second Qubit
qcircuit.h(1)

#draws a brilliant diagram of our circuit
qcircuit.draw(output='mpl')

#set our Simulators
S_simulator=Aer.backends(name='statevector_simulator')[0]
M_simulator=Aer.backends(name='qasm_simulator')[0]

job=execute(qcircuit, S_simulator)
ket=job.result().get_statevector()
#we get the amplitudes of each state in our qubit
#1/2|00> + 1/2|01> + 1/2|10> + 1/2|11>
#print(ket)


new_qcircuit = QuantumCircuit(qregister)
new_qcircuit.initialize(ket, qregister)
new_job=execute(qcircuit, S_simulator)
new_ket=new_job.result().get_statevector()
#print(new_ket)

cregister=ClassicalRegister(2)
qcircuit.add_register(cregister)

qcircuit.measure(qregister[0],cregister[0])
qcircuit.measure(qregister[1],cregister[1])


qcircuit.draw(output='mpl')


job=execute(qcircuit, M_simulator, shots=5000)
hist=job.result().get_counts()
#print(hist)

#Qiskit has awesome visualization tools!
from qiskit.visualization import plot_histogram
#plot_histogram(hist)

'''
SETUP TELEPORTATION
'''
#WE USE BARRIERS TO SEPERATE STEPS
import numpy as np

#ALICE has qubit 1
qr=QuantumRegister(3)
crz=ClassicalRegister(1)#where we get our z measurements
crx=ClassicalRegister(1)#where we get out x measurements

teleportation_circuit=QuantumCircuit(qr,crz,crx)

'''
STEP 5 comeback to top
'''
#create a random state for 1 qubit
psi=[1/np.sqrt(2),1/np.sqrt(2)]

from qiskit.extensions import Initialize
#make an initialization gate

init_gate = Initialize(psi)

teleportation_circuit.append(init_gate, [0])

'''
Step 1
'''
#Hadammard on second qubit
teleportation_circuit.h(1)
#CNOT second qubit is control and
#third qubit is target
teleportation_circuit.cx(1,2)
#make a barrier to seperate steps
teleportation_circuit.barrier()
'''
Step 2
'''
#CNOT first qubit is control and
#second is target
teleportation_circuit.cx(0,1)
#Hadammard on first qubit
teleportation_circuit.h(0)
#make a barrier to seperate steps
teleportation_circuit.barrier()

'''
Step 3
'''
#Alice measures q0 and q1 the first and second qubits
teleportation_circuit.measure(0,0)
teleportation_circuit.measure(1,1)

teleportation_circuit.barrier()
'''
Step 4
'''

#We apply X^j and Z^k depending on our measurements
#execute depending on state of classical register
teleportation_circuit.z(2).c_if(crz, 1)
teleportation_circuit.x(2).c_if(crx, 1)
#Our complete circuit
teleportation_circuit.draw(output='mpl')

'''
Now we have a circuit we can use!
'''

'''
Step 6
How do we check?
'''
#init|0> -> |psi>
# fine init_inverse to get init_inverse|psi> -> |0>
#defines inverse opperation for what you give it
inverse_init_gate=init_gate.gates_to_uncompute()
#puts our inverse gate on Bobs's qubit
teleportation_circuit.append(inverse_init_gate, [2])
#a register just for Bob
cr_result=ClassicalRegister(1)
teleportation_circuit.add_register(cr_result)
#Measure it!
teleportation_circuit.measure(2,2)



teleportation_circuit.draw(output='mpl')

job=execute(teleportation_circuit, M_simulator)
result=job.result().get_counts()
plot_histogram(result)



#states are written |cba> from left to right not right to left
# so qubit 3,2,1 in that order

#We can see that the last qubit the third one or qubit 2 if we count from 0
# is always in the 0 state no matter how we measure.
#The only way to get 0 is if it was in the initialized state before we disentangled Bob's qubit!! Voila it works!! 
