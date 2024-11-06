# https://github.com/Samikmalhotra/Quantum-Computing/blob/00427b7bdf518de5e50848f35745dd75c5fea8a6/hello-quantum.py
# %%
from qiskit.tools.monitor import job_monitor
from qiskit.providers import backend
import qiskit.tools.jupyter
from matplotlib import *
from qiskit import *
from matplotlib import *
from qiskit.visualization import plot_histogram

# %%
circuit = QuantumCircuit(2, 2)
circuit.draw(output='mpl')

# %%
circuit.h(0)
circuit.draw(output='mpl')

# %%
circuit.cx(0, 1)  # 0->control-qubit  1->target qubit
circuit.measure([0, 1], [0, 1])
circuit.draw(output='mpl')

# %%
simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, simulator).result()
plot_histogram(result.get_counts(circuit))

# %%
IBMQ.load_account()

# %%
provider = IBMQ.get_provider("ibm-q")
quantum_computer = provider.get_backend('ibmq_lima')

# %%
job = execute(circuit, backend=quantum_computer)

# %%
job_monitor(job)
# %%
quantum_result = job.result()
plot_histogram(quantum_result.get_counts(circuit))

# %%
