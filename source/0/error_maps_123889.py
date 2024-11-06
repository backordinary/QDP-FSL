# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/error_maps.py
from qiskit import Aer
from qiskit import IBMQ
from qiskit.visualization import plot_error_map, plot_gate_map
from read_config import get_api_key
from matplotlib import pyplot as plt

IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider()
backends = provider.backends(simulator=False)[0]

print("list of backends!!")
print(backends)
plot_error_map(backends)
plt.show()
plot_gate_map(backends)
plt.show()