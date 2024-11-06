# https://github.com/alu0101133201/QuantumMicroprograms/blob/4050e92f42e30da01a640a7ed2d0855f4ff928fb/src/QFT3Qubits.py
import numpy as np
from numpy import pi
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ, BasicAer, execute
from qiskit.visualization import plot_bloch_multivector

# Circuit which implements the QFT
qc = QuantumCircuit(3)

qc.h(2)
qc.cp(pi/2, 1, 2) 
qc.cp(pi/4, 0, 2) 
qc.h(1)
qc.cp(pi/2, 0, 1) 
qc.h(0)
qc.swap(0,2)
qc.draw()

# Input circuit with example number 5 (101)
inputqc = QuantumCircuit(3)
inputqc.x(0)
inputqc.x(2)

svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(inputqc)
entangled_state = svsim.run(qobj).result()
stateVector = entangled_state.get_statevector()
plot_bloch_multivector(stateVector)

# Build the final circuit
finalqc = inputqc + qc
finalqc.draw()

# Show the results of the QFT circuit with test input 
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(finalqc)
entangled_state = svsim.run(qobj).result()
stateVector = entangled_state.get_statevector()

plot_bloch_multivector(stateVector)