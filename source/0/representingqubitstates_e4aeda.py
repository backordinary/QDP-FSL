# https://github.com/Algorithmist-Girl/QuantumComputingConcepts_From_IBMTextbook/blob/de7810f9aa8c2a30fb0f1178e67d3be406069538/QuantumStates&Qubits/RepresentingQubitStates.py

# use statevectors to describe the state of the system
#  statevector = collectgion of numbers in a vector, where each element in the statevector tells the prob of finding
# something at a certain place

# |0> and |1> form an orthonormal basis ==> can represent any 2D vector with a combo of these 2 states
# qubit's statevector is generally described as a linear combo of both |0> and |1> = SUPERPOSITION

from qiskit.visualization import plot_histogram, plot_bloch_vector
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
from math import sqrt, pi

quantum_circuit = QuantumCircuit(1)

# can use initialize method to transform the qubit into any state!
# tell which qubits to initialize!


# definining initial state as |1> instead of |0>
initial_state = [0, 1]

# which qubits to operate on???
quantum_circuit.initialize(initial_state, 0)
print(quantum_circuit.draw())
# plt.show()

# use qiskit's simulator to view the resulting state of the qubit
simulator = Aer.get_backend('statevector_simulator')

# use execute to run the circuit!
result = execute(quantum_circuit, simulator).result()

# can now obtain statevector of resulting qubit after the execution!
statevectorFinal = result.get_statevector()
print(statevectorFinal)

# uses j to represent complex number i! (0j because 0 complex numbers!)
# 0 + 0j = 0 and 1 + 0j = 1

# measure qubit on quant comp and see result!
quantum_circuit.measure_all()
print(quantum_circuit.draw())
# plt.show()

#can also get the counts from the execute method!
numCountsForHisto= result.get_counts()
plot_histogram(numCountsForHisto)
plt.show()
#  of course 100% chance of measuring 1!


# put qubit into superposition now!
quantum_circuit = QuantumCircuit(1)
# can add complex part to initial_state, with real and imaginary parts to initial_state
# how much in the |0> space and how much in the |1> area ==> linear combination
initial_state = [1j/sqrt(2), 1/sqrt(2)]
quantum_circuit.initialize(initial_state, 0)
res = execute(quantum_circuit, simulator).result().get_counts()
plot_histogram(res)
plt.show()


# rules of measurement: can apply a rule to obtain the measurement
# implication of this rule: needs to be normalized!! ==> probabilities should add up to 1
# probabilities shoudl add to 1 ==> magnitude of statevector should be 1
# if vector is NOT normalized and we call initialized on it, error will be thrown!



# INTERESTING PROPERTY: the act of measuring does change the state of our qubits!!!
# this is equivalent to collapsing the state of the qubit
# so measurements should ONLY be used when we want to extract the output!!!