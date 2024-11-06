# https://github.com/n-ando/qiskit_tutorial/blob/f94ab09a5f3f1ea7bba22e180620f43ae44cecc2/bitop.py
#!/usr/bin/env python3
# 
# まずは必要になるpythonモジュールをすべてインポートしておく
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

print('notebook ready')

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.ry(np.pi / 2., 0)
circuit.x(0)
circuit.measure_all()
circuit.draw()

print('This curcuit has', circuit.num_qubits, 'qubits and', circuit.size(), 'operations')

