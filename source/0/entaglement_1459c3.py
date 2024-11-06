# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/entaglement.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

#
# -------------- cx circuit -----------------------
#

qc1 = QuantumCircuit(2)

# qc.cx(0,1) or qc.cx(0,1) alone would have no effect, with the state remaining |00>

# to change the state, we need to first flip the control qubit
qc1.x(1, 0)
qc1.cx(1, 0)

qc1.draw(output="mpl")
plt.show()


# this calculates what the state vector of our qubits would be
# after passing through the circuit 'qc1'
ket1 = Statevector(qc1)

# ket.draw("latex") writes down the state vector
# since it's the last line in the cell, the cell will display it as output
print(ket1)


#
# -------------- entaglement circuit -----------------------
#

qc2 = QuantumCircuit(2)

# this is how the two states are entagled
qc2.h(1)
qc2.cx(1, 0)

qc2.draw(output="mpl")
plt.show()

ket2 = Statevector(qc2)

print(ket2)


#
# -------------- phase kickback circuit -----------------------
#

qc3 = QuantumCircuit(2)

# these two alone have again no effect
qc3.h([0, 1])
qc3.cx(1, 0)

# now flip the target qubit from |+> to |-> using the single qubit z gate
qc3.z(0)
qc3.cx(1, 0)

qc3.draw(output="mpl")
plt.show()


ket3 = Statevector(qc3)

print(ket3)
