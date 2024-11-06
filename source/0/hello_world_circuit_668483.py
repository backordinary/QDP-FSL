# https://github.com/jazzyoverflow/Qiskit-Experimentation/blob/1a9b58390928dcc5c376230714167bce02c1d2ee/scripts/hello_world_circuit.py
# bring in necessary modules
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram


# number of quantum bits 
n_qbits = 8

# number of classical bits
n_bits = 8 

# create the quantum circuit
output_circuit = QuantumCircuit(n_qbits, n_bits)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(0, 0)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(1, 1)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(2, 2)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(3, 3)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(4, 4)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(5, 5)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(6, 6)
# NOTE: measure(qbit, outbit) measures result of qbit into outbit, can use ints since these are indexable in our circuit
output_circuit.measure(7, 7)

print(output_circuit.draw())



# execute circuit and save the observed results
# NOTE: qasm_simulator stands for quantum assembly language
# NOTE: get_backend is the environment you'd like to execute in 
observations = execute(circuit, Aer.get_backend('qasm_simulator')).result().get_counts()


print(plot_histogram(observations))



