# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/temp.py
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from scipy import stats
from plot import plot_hist
import numpy as np
from numpy import inf

IBMQ.load_accounts()
IBMQ.backends()

_MICROSECOND_UNITS = {'s': 1e6, 'ms': 1e3, 'µs': 1, 'us': 1, 'ns': 1e-3}
_GHZ_UNITS = {'Hz': 1e-9, 'KHz': 1e-6, 'MHz': 1e-3, 'GHz': 1, 'THz': 1e3}


def _check_for_item(lst, name):
	"""Search list for item with given name."""
	for item in lst:
		if item.name == name:
			item.value = 0

# device = IBMQ.get_backend('ibmq_16_melbourne')
# device = IBMQ.get_backend('ibmq_5_yorktown')
device = IBMQ.get_backend('ibmq_5_tenerife')
properties = device.properties()
coupling_map = device.configuration().coupling_map

values = []
for qubit, qubit_props in enumerate(properties.qubits):
	# Default values
	t1, t2, freq = inf, inf, inf

	# Get the readout error value
	t1_params = _check_for_item(qubit_props, 'T1')
	t2_params = _check_for_item(qubit_props, 'T2')
	freq_params = _check_for_item(qubit_props, 'frequency')

	# # Load values from parameters
	# if hasattr(t1_params, 'value'):
	#     t1 = t1_params.value
	#     if hasattr(t1_params, 'unit'):
	#         # Convert to micro seconds
	#         t1 *= _MICROSECOND_UNITS.get(t1_params.unit, 1)
	# if hasattr(t2_params, 'value'):
	#     t2 = t2_params.value
	#     if hasattr(t2_params, 'unit'):
	#         # Convert to micro seconds
	#         t2 *= _MICROSECOND_UNITS.get(t2_params.unit, 1)
	# if hasattr(t2_params, 'value'):
	#     freq = freq_params.value
	#     if hasattr(freq_params, 'unit'):
	#         # Convert to Gigahertz
	#         freq *= _GHZ_UNITS.get(freq_params.unit, 1)

	# # NOTE: T2 cannot be larged than 2 * T1 for a physical noise
	# # channel, however if a backend eroneously reports such a value we
	# # truncated it here:
	# t2 = min(2 * t1, t2)

	# values.append((t1, t2, freq))
for qubit, qubit_props in enumerate(properties.qubits):
	for item in qubit_props:
		if item.name == "frequency":
			print(item)

