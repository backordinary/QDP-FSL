# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuittransform/get_coupling_map_list.py
from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    converters,
    IBMQ)
from qiskit.transpiler import CouplingMap, Layout

provider = IBMQ.load_account()

#simulated_backend = provider.get_backend('ibmq_qasm_simulator')
simulated_backend = provider.get_backend('ibmqx5')
coupling_map_list = simulated_backend.configuration().coupling_map  # Get coupling map from backend
coupling_map = CouplingMap(coupling_map_list)
print(coupling_map_list)