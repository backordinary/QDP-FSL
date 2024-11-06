# https://github.com/rodrigopff/ibm_quantica/blob/a6d3d763134aeaad4ae4a272bebdb4a0693189d8/medida_X.py
# Run the code in this cell to see the widget
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from numpy import sqrt

# Create the X-measurement function:
def x_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc

initial_state = [1/sqrt(2), -1/sqrt(2)]
# Initialize our qubit and measure it
qc = QuantumCircuit(1,1)
qc.initialize(initial_state, 0)
x_measurement(qc, 0, 0)  # measure qubit 0 to classical bit 0
#qc.draw('text')
qc.draw('mpl')
#print(qc.draw('text'))
plt.show()
