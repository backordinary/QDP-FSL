# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuittransform/test.py
from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    transpile,
    converters,
    IBMQ)
from qiskit.transpiler import CouplingMap, Layout
from qiskit import QuantumRegister
from qiskit.test.mock import FakeTokyo
import datetime
import time


from qiskit.dagcircuit import dagcircuit
import matplotlib.pyplot as plt
import pydot

method_AG = ['IBM QX20']

qr = QuantumRegister(20)

input_dict = {
    qr[0]: 4,
    qr[1]: 3,
    qr[2]: 2,
    qr[3]: 1,
    qr[4]: 0
}
layout_dict = Layout()
layout_dict = layout_dict.from_dict(input_dict=input_dict)
filename = "./qasm/qft_10.qasm"
circuit = QuantumCircuit.from_qasm_file(filename)
circuit.measure_all()

provider = IBMQ.load_account()
# backend = provider.get_backend('ibmq_16_melbourne')
backend = FakeTokyo()
starttime = time.time()
circ = transpile(circuit, backend, optimization_level=0, initial_layout=[4, 11, 10, 5, 8, 9, 3, 12, 2, 6, 13, 1, 7, 14, 15, 0])
print('Optimization Level {}'.format(0))
print('Depth:', circ.depth())
print('Gate counts:', circ.count_ops())
# dag = converters.circuit_to_dag(circ)
# dag.draw()
# plt.show()
print()
endtime = time.time()
print (endtime - starttime)
starttime = time.time()
circ = transpile(circuit, backend, optimization_level=1, initial_layout=[4, 11, 10, 5, 8, 9, 3, 12, 2, 6, 13, 1, 7, 14, 15, 0])
print('Optimization Level {}'.format(1))
print('Depth:', circ.depth())
print('Gate counts:', circ.count_ops())
# dag = converters.circuit_to_dag(circ)
# dag.draw()
# plt.show()
print()
endtime = time.time()
print (endtime - starttime)
starttime = time.time()
circ = transpile(circuit, backend, optimization_level=2, initial_layout=[4, 11, 10, 5, 8, 9, 3, 12, 2, 6, 13, 1, 7, 14, 15, 0])
print('Optimization Level {}'.format(2))
print('Depth:', circ.depth())
print('Gate counts:', circ.count_ops())
# dag = converters.circuit_to_dag(circ)
# dag.draw()
# plt.show()
print()
endtime = time.time()
print (endtime - starttime)
starttime = time.time()
circ = transpile(circuit, backend, optimization_level=3, initial_layout=[4, 11, 10, 5, 8, 9, 3, 12, 2, 6, 13, 1, 7, 14, 15, 0])
print('Optimization Level {}'.format(3))
print('Depth:', circ.depth())
print('Gate counts:', circ.count_ops())
# dag = converters.circuit_to_dag(circ)
# dag.draw()
# plt.show()
print()
endtime = time.time()
print (endtime - starttime)