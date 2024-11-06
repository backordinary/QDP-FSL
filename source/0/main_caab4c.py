# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/main.py
import qiskit as q
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
# from qiskit.tools.visualization import plot_bloch_sphere
from qiskit.visualization import plot_histogram, matplotlib
import matplotlib.pyplot as plt
from utils import dotdict

# %matplotlib inline

config = dotdict()
config.location = "local"  # local or remote
config.sim_backend = "qasm_simulator"
config.backend = ""
config.shots = 100

if config.location == "local":
    # Run locally.
    backend = q.Aer.get_backend(config.sim_backend)
else:
    # Run on backend.
    IBMQ.save_account(open("token.txt", "r").read())
    IBMQ.load_account()
    provider = IBMQ.get_provider("ibm-q")
    for backend in provider.backends():
        try:
            qubit_count = len(backend.properties().qubits)
        except:
            qubit_count = "simulated"
        print(f"{backend.name():35} has {backend.status().pending_jobs:8} queued and {qubit_count:12} qubits")
    backend = ""

# Build the quantum circuit.
circuit = q.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all(True)
circuit.draw()
print(circuit)
# circuit.draw(output="mpl")
# plt.show()

# Build the job.
job = q.execute(circuit, backend=backend, shots=config.shots)
job_monitor(job)

# Results.
result = job.result()
counts = result.get_counts(circuit)
print(counts)

# Visualize.
plot_histogram([counts])
plt.show()
