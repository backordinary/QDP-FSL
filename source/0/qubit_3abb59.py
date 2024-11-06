# https://github.com/i0Ek3/Xperimental/blob/fdb0962a6e5ae3339f8af38894b34b464f56ef66/Quantum/ibm/qubit.py
from math import sqrt, pi
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.visualization import plot_bloch_multivector

# change alpha beta as you want
alpha = 1/sqrt(2)
beta = 1j/sqrt(2)

# 2D complex vector representing the quantum state
vector = [alpha, beta]

# create a quantum circuit with only one qubit
qc = QuantumCircuit(1)
# initialzie the qubit as the vector we defined
# arg 0 is the index of the only qubit we created
qc.initialize(vector, 0)


# simulate the state
# note that statevector_simulator is used for calculating statevector (Quantum State)
backend = BasicAer.get_backend('statevector_simulator')
result = execute(qc, backend).result()
state = result.get_statevector(qc)

# plot the state on a Bloch sphere
plot_bloch_multivector(state)
