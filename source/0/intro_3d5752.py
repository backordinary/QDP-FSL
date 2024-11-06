# https://github.com/Algorithmist-Girl/QuantumComputingConcepts_From_IBMTextbook/blob/de7810f9aa8c2a30fb0f1178e67d3be406069538/QuantumStates&Qubits/Intro.py
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, execute, Aer
import matplotlib.pyplot as plt

#  operations on qubits using gates!
# creating a circuit ==> 3 steps, encode the input, perform the compuation, and extract output!


# need to define the # of qubits in circuit and number of output bits!
NumberQubits = 10
NumberOutputBits = 10
quantum_circuit = QuantumCircuit(NumberQubits, NumberOutputBits)

# use measure op to get the result of the circuit!
# each measurement==> tells specific qubit to give an output to specific output bit!

# add measure opp to all of the qubits!, numbered 0 to 9
for ct in range(NumberOutputBits):
    quantum_circuit.measure(ct, ct)
# telling qubit ct to write to bit j!!

print(quantum_circuit.draw())

# qubits initialized to give 0 output
# because this is a probability distribution, can plot in histogram! (since there's superposition,etc.)

res = execute(quantum_circuit, Aer.get_backend('qasm_simulator')).result().get_counts()
plot_histogram(res)
plt.show()
# quant comps can have some randomness in their results ==> histogram is a good result!


