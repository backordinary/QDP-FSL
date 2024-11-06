# https://github.com/Samikmalhotra/Quantum-Computing/blob/00427b7bdf518de5e50848f35745dd75c5fea8a6/Quantum-Teleporation/quantum_teleportation.py
# %%
from qiskit import *
from qiskit import circuit
from qiskit.visualization import plot_histogram
from matplotlib import *

# %%
circuit = QuantumCircuit(3, 3)

circuit.x(0)

circuit.barrier()

circuit.h(1)
circuit.cx(1, 2)

circuit.barrier()

circuit.draw(output='mpl')

# %%
circuit.cx(0, 1)
circuit.h(0)

circuit.barrier()
circuit.draw(output='mpl')

# %%
circuit.measure([0, 1], [0, 1])
circuit.barrier()

circuit.draw(output='mpl')

# %%
circuit.cx(1, 2)
circuit.cz(0, 2)

circuit.measure([2], [2])
circuit.draw(output='mpl')

# %%
simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator, shots=1024).result()
plot_histogram(result.get_counts(circuit))

# %%
