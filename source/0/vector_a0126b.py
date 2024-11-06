# https://github.com/rodrigopff/ibm_quantica/blob/a6d3d763134aeaad4ae4a272bebdb4a0693189d8/vector.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt


qc = QuantumCircuit(2)

# This calculates what the state vector of our qubits would be
# after passing through the circuit 'qc'
ket = Statevector(qc)

# The code below writes down the state vector.
# Since it's the last line in the cell, the cell will display it as output
print(ket.draw('text'))
plt.show()