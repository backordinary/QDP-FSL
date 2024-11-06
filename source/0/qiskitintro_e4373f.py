# https://github.com/ElektrosStulpas/QuantumComputingProjectVU/blob/af7564d04e59db091ef0d199fc4179474bc397b7/QiskitIntro.py
#intro to qiskit from https://towardsdatascience.com/what-is-quantum-entanglement-anyway-4ea97df4bb0e
from matplotlib.pyplot import title
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib as mpl

M_simulator=Aer.backends(name='qasm_simulator')[0]

qreg = QuantumRegister(2) # qreg is filled with two qubits 
creg = ClassicalRegister(2) # creg is filled with two classical bits

entangler=QuantumCircuit(qreg, creg) # we put our qreg and creg together to make our Quantum Circuit, called entangler here.

entangler.h(0) # puts first qubit through hadamard
entangler.cx(0, 1) # applies CNOT with first cubit as control and second qubit as target

entangler.measure(0, 0) # first choose quantum bit and measure it into the chosen classical bit
entangler.measure(1, 1) # measure second qubit into second cbit

fig = entangler.draw(output='mpl') 
fig.show()

job = execute(entangler, M_simulator) # executes numerous measurements given a circuit and a backend simulator
hist = job.result().get_counts()
fig = plot_histogram(data=hist, title="Maximally entangled state")
fig.show()