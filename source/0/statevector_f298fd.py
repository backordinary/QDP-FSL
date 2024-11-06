# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/statevector.py
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.visualization.gate_map import plot_coupling_map, plot_gate_map
from qiskit.visualization.state_visualization import plot_bloch_vector, plot_state_city, plot_state_paulivec
circuit = QuantumCircuit(1,1)
circuit.x(0)
simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
print(statevector)

simulator2 = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend= simulator2).result()
unitary = result.get_unitary()
print(unitary)