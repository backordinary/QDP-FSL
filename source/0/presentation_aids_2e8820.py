# https://github.com/andrijapau/qml-thesis-2022/blob/8b348fd17c1daf8b5b7de037577aad056a4eed8b/Presentation/presentation_aids.py
# Import Tools
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import visualize_transition
import matplotlib.pyplot as plt
from qiskit import IBMQ
from qiskit.visualization import plot_histogram

IBMQ.load_account()
provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

# Create a Quantum Circuit to Visualize
num_of_qubits = 1
num_of_cbits = 1
quantumCircuit = QuantumCircuit(num_of_qubits, num_of_cbits)

# Apply Hadamard
quantumCircuit.h(0)

# Measure
quantumCircuit.measure(0, 0)

# Draw
quantumCircuit.draw('mpl', filename='hadamard.png', style={'name': 'bw', 'dpi': 350})
plt.show()

quantumCircuit.qasm(filename='hadamard_qasm')

job = execute(
    quantumCircuit,
    backend=provider.get_backend("ibmq_armonk"),
    shots=10000
)

plot_histogram(job.result().get_counts(), filename='hadamard.png', color='black',
               title="Hadamard Gate Results")
plt.show()
# Visualize
# visualize_transition(quantumCircuit, saveas='./hadamard.gif')
