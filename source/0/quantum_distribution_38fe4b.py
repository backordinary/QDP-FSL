# https://github.com/AlexanderKosmehl/qiskit-normal-distribution/blob/3e04b9d3295a629765812d8867c2df384b4c1108/quantum_distribution.py
from qiskit import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.aqua.components.uncertainty_models import NormalDistribution
from qiskit.providers.aer import AerSimulator
sim = AerSimulator()
from collections import OrderedDict
from numpy.core.fromnumeric import std
from numpy.lib.function_base import average

# Variable Setup
mu, sigma = 560, 200
num_qubits = 5
low = (mu - 3 * sigma)
high = (mu + 3 * sigma)

# Generate Circuit
q = QuantumRegister(num_qubits, 'q')
c = ClassicalRegister(num_qubits, 'c')
circuit = QuantumCircuit(q, c)

# Build Distribution Circuit
normal = NormalDistribution(num_target_qubits=num_qubits, mu=mu, sigma=sigma, low=low, high=high)
normal.build(circuit, q)
circuit.measure(q, c)

# Execute Simulation
job = sim.run(circuit)
counts = job.result().get_counts()

# Transform Results into correct Distribution
int_counts = {int(key, 2):value for (key, value) in counts.items()}
mapped_counts = {((high - low) / (2**num_qubits - 1) * number + low):occurences for number, occurences in int_counts.items()}
ordered_counts = OrderedDict(sorted(mapped_counts.items()))

# Transform Results into list to draw samples from
sample_list = []
for number, occurences in ordered_counts.items():
  for _ in range(occurences):
    sample_list.append(number)

# Check results
print(f"Average: {average(sample_list)}")
print(f"Std: {std(sample_list)}")