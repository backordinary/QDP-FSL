# https://github.com/qcware/qiskit_qcware/blob/975bb6e1d0f9a5b306f1d781b45c933dfdae14ff/tests/performance/qiskit_profile.py
import numpy as np
import qiskit
from qiskit.optimization.applications.ising.common import random_graph
from qiskit.optimization.applications.ising import vertex_cover
from qiskit.aqua.algorithms import QAOA
import qiskit
from typing import List
import cProfile
from qcware_transpile.translations.qiskit.to_quasar import translate

np.random.seed(123)
num_nodes = 6
Max_circuit_width = 20
# Our circuit widths in number of qubits
Circuit_widths = list(range(4, Max_circuit_width + 1, 2))
Num_parameters = [1] * len(Circuit_widths)
Parameters = [list(np.random.random(2 * x) * np.pi) for x in Num_parameters]


def create_qiskit_circuit(num_nodes: int,
                          params: List[float],
                          edge_prob=0.8,
                          weight_range=10) -> qiskit.QuantumCircuit:
    # print(locals())
    np.random.seed(123)
    w = random_graph(num_nodes, edge_prob=edge_prob, weight_range=weight_range)
    qubit_op, offset = vertex_cover.get_operator(w)
    qaoa = QAOA(qubit_op, p=int(len(params) / 2))
    return qaoa.var_form.construct_circuit(params)

circuit_width = 7
qiskit_circuit = create_qiskit_circuit(Circuit_widths[circuit_width],
                                       Parameters[circuit_width])
print(f"{qiskit_circuit.num_qubits} qubits")
# translate once to load any missing imports
# quasar_circuit = translate(qiskit_circuit)
from qiskit_qcware import QcwareProvider
backend = QcwareProvider().get_backend('forge_measurement')
data = qiskit.execute(qiskit_circuit, backend).result().data()
cProfile.run("sv = qiskit.execute(qiskit_circuit, backend).result().data()", 'test.prof')
