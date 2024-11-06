# https://github.com/StevenOwino/Commutative_Operators/blob/f80db1cbb27cd309fb069feb14c9bbc621e41458/Commutative_Operators/Entangled%20States.py
#Quantum Circuits with Qiskit : Controlled-NOT Gate
#An entangled state is a mixed state(its statistics exhibit both classical and quantum form of correlation between probabilistic distribution and quantum wavefunctions)

from qiskit import QuantumCircuit
qc = QuantumCircuit(3, 3)
# measure qubits 0, 1 & 2 to classical bits 0, 1 & 2 respectively
qc.measure([0,1,2], [0,1,2])
qc.draw()


#CX Gate: Two qubits as 'control' and 'target'
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
qc = QuantumCircuit(2)
# This calculates what the state vector of our qubits would be
# after passing through the circuit 'qc'
ket = Statevector(qc)
# The code below writes down the state vector.
# Since it's the last line in the cell, the cell will display it as output
ket.draw() # Output State: |00>

qc.cx(0,1)
ket = Statevector(qc)
ket.draw() #Output State remains |00>

#Control and target reversed, output still remains |00>
qc.cx(1,0)
ket = Statevector(qc)
ket.draw()

#When the control is 0. the other qubit is unchanged, when the control is 1, the other qubit is flipped.

#Flipping the control qubit to |1> state
qc.x(1,0)
ket = Statevector(qc)
ket.draw() #Output state is |10>

#Flipping the target qubit to |1> state
qc.cx(1,0)
ket = Statevector(qc)
ket.draw() #Output state is |11>

# Let's create an entagled state
qc = QuantumCircuit(2)
qc.h(1)
ket = Statevector(qc)
ket.draw() #Output is sqrt2/2 |00> + Sqrt2/2 |10>

#Now if we apply the cx, it will act in parallel on the two states. It will leave the |00> state unchanged, since the control qubit there is in state |00> . But |10>  has the control in state |1> , so the target is flipped to make the |11> state.

qc.cx(1,0)
ket = Statevector(qc)
ket.draw() 
# Output Sqrt2/2 |00> + Sqrt2/2 |11>: This is the entangled state |Phi+> created by the cx 
# gate

#However, let's now flip the target qubit from |+ > to |- > using the single qubit z gate
# Phase kickback (Diffuser): As frequency(Omega) goes to infinity, phase angle shifts from 90 degrees to 0. 
#This is state of a circuit during a voltage drop/current draw/impedance
qc.z(0)
ket = Statevector(qc)
ket.draw() #Output 
#If we do the cx now, we will see an effect. It flips the control qubit to |- > as well.
qc.cx(1,0)
ket = Statevector(qc)
ket.draw()







































