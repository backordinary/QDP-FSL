# https://github.com/i0Ek3/Xperimental/blob/fdb0962a6e5ae3339f8af38894b34b464f56ef66/Quantum/ibm/hadamard.py
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.visualization import plot_histogram

# create a quantum circuit with one qubit and one bit
qc = QuantumCircuit(1,1)
# apply hadamard gate on qubit 0
qc.h(0)
# measure qubit 0 to bit 0
qc.measure(0,0)
# plot the circuit
qc.draw('mpl')
