# https://github.com/talm95/QuantumComputing/blob/8e63961784806d863244ef3aab831af0f6bc668b/GroversAlgorithm/complete_algorithm.py
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from s_reflection import reflect_around_s_operation
import matplotlib.pyplot as plt


def grover_algorithm(grover_circuit, oracle, s_reflection, iterations, number_of_qubits):
    for qubit in range(number_of_qubits):
        grover_circuit.h(qubit)
    for iteration in range(iterations):
        grover_circuit.append(oracle, range(number_of_qubits))
        grover_circuit.append(s_reflection, range(number_of_qubits))

    for qubit in range(number_of_qubits):
        grover_circuit.measure(qubit, qubit)
    return grover_circuit


# solving for w = 10
oracle = QuantumCircuit(2, name='oracle')
oracle.x(0)
oracle.cz(0, 1)
oracle.x(0)

s_reflection = reflect_around_s_operation(2)
grover_circuit = QuantumCircuit(2, 2)
grover_algorithm(grover_circuit, oracle, s_reflection, 1, 2)
grover_circuit.draw('mpl')
plt.show()


qasm_sim = Aer.get_backend('qasm_simulator')
shots = 2048
t_qpe = transpile(grover_circuit, qasm_sim)
qobj = assemble(t_qpe, shots=shots)
results = qasm_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()
