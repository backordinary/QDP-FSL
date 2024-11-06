# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/testfiles/dist_map.py
import math
import os

from qiskit import QuantumCircuit, converters, QuantumRegister, transpile, IBMQ
from qiskit.circuit import Qubit
from qiskit.test.mock import FakeMelbourne

# import sys
from qiskit.transpiler import Layout, CouplingMap


def dist_layout(current_layout, target_layout, coupling_map, qubit_num, qr):
    dist = 0
    for i in range(qubit_num):
        dist += coupling_map.distance(current_layout.get_virtual_bits()[Qubit(qr, i)],
                                         target_layout.get_virtual_bits()[Qubit(qr, i)])
    return dist


if __name__ == '__main__':
    backend1 = FakeMelbourne()
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmqx2')
    coupling = backend.configuration().coupling_map
    qubits = backend.configuration().n_qubits
    qr = QuantumRegister(qubits)
    print(coupling)
    print(qubits)
    layout = Layout.from_intlist([0, 1, 2, 3, 4], qr)
    layout2 = Layout.from_intlist([2, 3, 4, 0, 1], qr)
    print(layout)
    print(layout2)
    coupling_map = CouplingMap(coupling)
    dist = dist_layout(layout, layout2, coupling_map, qubits, qr)
    print(dist)

