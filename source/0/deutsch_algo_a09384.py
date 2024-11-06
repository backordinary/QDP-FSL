# https://github.com/Samikmalhotra/Quantum-Computing/blob/00427b7bdf518de5e50848f35745dd75c5fea8a6/Deutsch-Algorithm/deutsch-algo.py
# %%
from qiskit import *
from qiskit.providers import ibmq
from qiskit.tools.visualization import plot_histogram
from matplotlib import *

# %%
circuit = QuantumCircuit(2, 1)

# %%
circuit.h(0)
circuit.x(1)
circuit.h(1)
circuit.barrier()
circuit.draw(output='mpl')

# %%
circuit.cx(0, 1)
circuit.barrier()
circuit.h(0)
circuit.barrier()
circuit.draw(output='mpl')

# %%
circuit.measure(0, 0)
circuit.draw(output='mpl')

# %%
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=backend, shots=1024).result()
counts = result.get_counts(circuit)

plot_histogram([counts])

# %%
# Real Quantum Computer
IBMQ.load_account()
# %%
provider = IBMQ.get_provider("ibm-q")
provider.backends()

# %%
for backend in provider.backends():
    try:
        qubit_count = len(backend.properties().qubits)
    except:
        qubit_count = "simulated"
    print(f"{backend.name()} : {backend.status().pending_jobs} pending jobs & {qubit_count} qubits")
# %%
quantum_computer = provider.get_backend('ibmq_belem')

# %%
quantum_result = execute(
    circuit, backend=quantum_computer, shots=1024).result()

# %%
quantum_counts = quantum_result.get_counts(circuit)
plot_histogram([quantum_counts])

# %%
