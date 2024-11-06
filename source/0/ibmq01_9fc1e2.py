# https://github.com/n-ando/qiskit_tutorial/blob/23e7df45614251f1347b64087cac84723dacdee6/ibmq01/ibmq01.py
#!/usr/bin/env python3
# https://utokyo-icepp.github.io/qc-workbook/chsh_inequality.html#id66
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

IBMQ.load_account()

# IBMQプロバイダ（実機へのアクセスを管理するオブジェクト）
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

# バックエンド（実機）のうち量子ビット数2個以上のもののリストをプロバイダから取得し、一番空いているものを選ぶ
backend_filter = lambda b: (not b.configuration().simulator) and (b.configuration().n_qubits >= 2) and b.status().operational
backend = least_busy(provider.backends(filters=backend_filter))

print('Jobs will run on', backend.name())

shots = 8192

job = execute(circuits, backend=backend, shots=shots)

job_monitor(job, interval=2)

result = job.result()

counts = []
for circuit in circuits:
    c = result.get_counts(circuit)
    counts.append(c)
    
print(counts)

for c in counts:
    ax = plt.figure().add_subplot()
    plot_histogram(c, ax=ax)

C = []
for c in counts:
    C.append((c['00'] + c['11'] - c['01'] - c['10']) / shots)
    
S = C[0] - C[1] + C[2] + C[3]

print('C:', C)
print('S =', S)
if S > 2.:
    print('Yes, we are using a quantum computer!')
    