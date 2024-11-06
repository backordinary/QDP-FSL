# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/testfiles/profiling_with_depth.py
import copy
import json
import math
import os
from qiskit import QuantumCircuit, QuantumRegister, converters, dagcircuit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

filename = '../examples/test.qasm'
qr = QuantumRegister(5)
profiling_array = np.zeros((5, 5))
profiling_array_depth = np.zeros((5, 5))
circuit = QuantumCircuit.from_qasm_file(filename)
dag = converters.circuit_to_dag(circuit)

# delete the single-qubit gates in dag
for node in dag.gate_nodes():
    if node.name != 'cx':
        dag.remove_op_node(node)

dag_copy = copy.deepcopy(dag)
depth = dag.depth()
front_layer = dag_copy.front_layer()
for node in dag.gate_nodes():
    profiling_array[node.qargs[0].index][node.qargs[1].index] += 1

"""
dag_copy的删除存在问题:
方案：
1. 继续调试dag和copy的问题
2. 考虑直接用原本的dag，或者重新生成一个dag
3. 重新生成一个dag，拓扑排序生成front layer
"""

#
for item in dag.gate_nodes():
    dag.remove_op_node(item)


for i in range(depth):
    print(front_layer)
    print(profiling_array_depth)
    for node in front_layer:
        profiling_array_depth[node.qargs[0].index][node.qargs[1].index] += (1 - i/depth)
        dag.remove_op_node(node)
    front_layer = dag.front_layer()

dag.draw()
plt.show()
print(profiling_array)
print(profiling_array_depth)


