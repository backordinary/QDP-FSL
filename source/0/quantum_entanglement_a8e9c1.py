# https://github.com/tsrobinson/quantum/blob/927f0cc7d46029f5ab2bbb926416d04fc3adfc31/quantum_entanglement.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Product statevectors
qc = QuantumCircuit(2)
ket = Statevector(qc)
ket.draw()

qc.cx(0,1)
ket = Statevector(qc)
ket.draw()

qc.x(1,0)
ket = Statevector(qc)
ket.draw()

qc.cx(1,0)
ket = Statevector(qc)
ket.draw()


# Entangled statevectors
qc = QuantumCircuit(2)
qc.h(1) # The second qubit is now in superposition
ket = Statevector(qc)
ket.draw()

qc.cx(1,0) # So here, since the second qubit (in superposition) is the control, then the target will flip with some probability as defined in the statevector
ket = Statevector(qc)
ket.draw()

# and crucially, there's no way to describe this as a sequence of two single qubit states

# Entanglement is necessary for a quantum system to have an advantage over classic computing



