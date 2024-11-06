# https://github.com/andrijapau/qml-thesis-2022/blob/5cba4b1dd156d53707f1cb2d7c835ec78e923432/misc-circuits/bell_state.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
import os
import matplotlib.pyplot as plt

num_of_qubits = 2
num_of_classical_bits = 2

q_reg = QuantumRegister(num_of_qubits)
c_reg = ClassicalRegister(num_of_classical_bits)

circuit = QuantumCircuit(q_reg, c_reg)

circuit.h(q_reg[0])
circuit.cx(q_reg[0], q_reg[1])

circuit.measure(q_reg, c_reg)

os.chdir('../circuit-photos')
path = os.getcwd()
circuit.draw('mpl', filename=path + '/bell_state_circuit.png', style={'name': 'bw', 'dpi': 350})
plt.show()

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
job = execute(circuit, backend=provider.get_backend("ibmq_qasm_simulator"), shots=10000)
os.chdir('../results-photos')
path = os.getcwd()
plot_histogram(job.result().get_counts(), filename=path + 'bell_state_pair_results.png', color='black',
               title="Bell's Pair Measurement Outcome")
plt.show()
