# https://github.com/kevinab107/q-algos/blob/56cb8471e2b334af8f94256b51eba9fa128ea769/test_qiskit.py
u  #%%
import numpy as np
from qiskit import *
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit import Aer
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor

from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP, L_BFGS_B
from qiskit.aqua.algorithms.adaptive import QAOA
from qiskit import BasicAer

from qiskit.aqua.algorithms import VQE
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

#%%
IBMQ.load_account()
provider = IBMQ.get_provider(group="open")
provider.backends()
provider.backends(
    filters=lambda x: x.configuration().n_qubits >= 10
    and not x.configuration().simulator
)
# %%
qc = QuantumCircuit(1)
qc.h(0)
# qc.cx(0, 1)
# qc.cx(1, 2)
qc.measure_all()
qc.draw()

# plot_state_city(outputstate)

# meas = QuantumCircuit(3, 3)
# meas.barrier(range(3))
# qc = meas + circ
# qc.measure(range(2), [0, 2])
# qc.draw()

simulator = Aer.get_backend("qasm_simulator")
# backend = provider.get_backend("ibmq_16_melbourne")
meas_job = execute(qc, simulator, shots=10000)
result_sim = meas_job.result()
counts_sim = result_sim.get_counts()

# quantum_job = execute(qc, backend)
# job_monitor(quantum_job)
# quantum_result = quantum_job.result()
# counts_quantum = quantum_result.get_counts()

plot_histogram([counts_sim], legend=["simulator"])


# %%
