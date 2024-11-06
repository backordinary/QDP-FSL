# https://github.com/BQSKit/qgo/blob/8afe200d60ffe4c9afa5dc49ead00a6d31a7183c/example/qgo_qaoa_q5.py
from qgo import qgo, CouplingMapAllToAll

from qiskit import QuantumCircuit
import numpy as np

coupling_map = CouplingMapAllToAll(5)
qc = QuantumCircuit.from_qasm_file('circuits/qaoa_q5_r2.qasm')
qgo_circ = qgo(circ=qc, coupling_map=coupling_map, reopt=True, reopt_single=True, syn_tool='leap', verbosity=2, b_size=3, sc_threshold=1e-10, name='qaoa_q5_b3', block_details=True)
qgo_circ.run()
print(qgo_circ.new_circuit.qasm())
