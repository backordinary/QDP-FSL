# https://github.com/n-ando/qiskit_tutorial/blob/f94ab09a5f3f1ea7bba22e180620f43ae44cecc2/chsh.py
#!/usr/bin/env python3
# 
# まずは必要になるpythonモジュールをすべてインポートしておく
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

circuits = []
circuit = QuantumCircuit(2, name='circuit{}'.format(0))
circuit.h(0)
circuit.cx(0, 1)
circuits.append(circuit)
circuit = QuantumCircuit(2, name='circuit{}'.format(1))
circuit.h(0)
circuit.cx(0, 1)
circuits.append(circuit)
circuit = QuantumCircuit(2, name='circuit{}'.format(2))
circuit.h(0)
circuit.cx(0, 1)
circuits.append(circuit)
circuit = QuantumCircuit(2, name='circuit{}'.format(3))
circuit.h(0)
circuit.cx(0, 1)
circuits.append(circuit)

circuits[0].ry(-np.pi / 4., 1)
circuits[1].ry(-3. * np.pi / 4., 1)
circuits[2].ry(-np.pi / 4., 1)
circuits[3].ry(-3. * np.pi / 4., 1)

circuits[2].ry(-np.pi / 2., 0)
circuits[3].ry(-np.pi / 2., 0)

for circuit in circuits:
    circuit.measure_all()

# draw()にmatplotlibのaxesオブジェクトを渡すと、そこに描画してくれる
# 一つのノートブックセルで複数プロットしたい時などに便利
for circuit in circuits:
    ax = plt.figure().add_subplot()
    circuit.draw('mpl', ax=ax)
