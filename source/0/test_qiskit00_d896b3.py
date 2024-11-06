# https://github.com/KennethGrace/intro_to_qiskit/blob/7d67933c4ed53548df8fa9976b0183b1665199aa/test_qiskit00.py
# %%
from qiskit.tools.visualization import plot_histogram
from qiskit import *
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_bloch_multivector
# %%
circuit = QuantumCircuit(1, 1)
circuit.x(0)
# %%
simulator = Aer.get_backend('unitary_simulator')
# %%
result = execute(circuit, backend=simulator).result()
# %%
unitary = result.get_unitary()
# %%
print(unitary)
# %%
circuit.draw(output='mpl')

# %%
circuit.measure([0], [0])


# %%
backend = Aer.get_backend('qasm_simulator')
# %%
result = execute(circuit, backend=backend, shots=1024).result()
# %%
counts = result.get_counts()
# %%
plot_histogram(counts)


# %%
