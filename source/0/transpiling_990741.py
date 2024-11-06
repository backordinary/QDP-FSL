# https://github.com/kevinab107/q-algos/blob/56cb8471e2b334af8f94256b51eba9fa128ea769/transpiling.py
#%%
import numpy as np
from qiskit import *
from qiskit.visualization import plot_state_city, plot_histogram, plot_circuit_layout, plot_gate_map
from qiskit import Aer
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor

#%%
IBMQ.load_account()
provider = IBMQ.get_provider(group= 'open')
backend = provider.get_backend('ibmq_16_melbourne')
#%%

qc = QuantumCircuit(2,1)
qc.h(0)
qc.x(1)
qc.cu1(np.pi/4,0,1)
qc.h(0)
qc.measure([0],[0])
qc.decompose().draw(output='mpl')


# %%
new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
plot_circuit_layout(new_circ_lv3, backend)
