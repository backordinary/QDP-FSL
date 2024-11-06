# https://github.com/qifanyyy/JupyterNotebook/blob/20d8df6172906337f81583dabb841d66b8f31857/new_algs/Number%2Btheoretic%2Balgorithms/Shor's%2Balgorithm/qftBenchmark.py
from qiskit.circuit.library import QFT
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
import simulation as sim
import argparse
import matplotlib.pyplot as plt
#args = sim.process_command()
x = []
gt_lis = []
ct_lis = []
for i in range(1, 29):
    x.append(i)
    qr = QuantumRegister(i)
    qc = QuantumCircuit(qr)
    gate = QFT(i)
    qc.append(gate, qargs=qr[:])
    qc.measure_all()
    gt_lis.append(sim.gpuSim(qc))
    ct_lis.append(sim.cpuSim(qc))

plt.plot(x, gt_lis,label='GPU')
plt.plot(x, ct_lis,label='CPU')
plt.legend()
plt.savefig('qftbenchmark.png')
