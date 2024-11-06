# https://github.com/Samikmalhotra/Quantum-Computing/blob/00427b7bdf518de5e50848f35745dd75c5fea8a6/simulators&providers.py
# %%
from qiskit import *
from qiskit.circuit.quantumregister import Qubit
from qiskit.providers import provider
IBMQ.save_account(open("ibmapi.txt", "r").read())
IBMQ.load_account()

# %%
Aer.backends()

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
