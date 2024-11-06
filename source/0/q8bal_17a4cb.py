# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/q8bal.py
import qiskit
from PIL import Image
from qiskit import IBMQ
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

# set up Quantum Register and Classical Register for 3 qubits
from qiskit.tools.visualization import matplotlib_circuit_drawer, plot_histogram

q = QuantumRegister(3)
c = ClassicalRegister(3)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)
qc.h(q) # create superposition


image = matplotlib_circuit_drawer(qc)
# Image._show(image)

qc.measure(q, c)
image = matplotlib_circuit_drawer(qc)
# Image._show(image)


qiskit.IBMQ.load_accounts()
backend = qiskit.backends.ibmq.least_busy(qiskit.IBMQ.backends(simulator=True))
print("We'll use the least busy device:", backend.name())
job = qiskit.execute(qc, backend)
stats = job.result().get_counts(qc)

print(stats)
plot_histogram(stats)

# for key in stats.keys():
#         state = key
#         print(state)