# https://github.com/PedruuH/Computacao-Quantica/blob/c39194368dbb02ebbafc9858904bad958f648e94/qft.py
# Requisitos:
# pip install qiskit
# pip install pylatexenc

import matplotlib.pyplot as plt
import qiskit as qk
from qiskit.tools.visualization import plot_histogram # para mostrar resultados
from qiskit.tools.monitor import job_monitor # para monitorar jobs nos computadores quânticos reais
from qiskit.circuit.library import QFT

# print(qk.__qiskit_version__) # para imprimir a versão do qiskit

from apitoken import apitoken # onde seu "token" está salvo

qr = qk.QuantumRegister(5)
cr = qk.ClassicalRegister(5)
circuit = qk.QuantumCircuit(qr, cr)
circuit.x(qr[4])
circuit.x(qr[2])
circuit.x(qr[0])
circuit += QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')
circuit.measure(qr,cr)
circuit.draw(output='mpl')
plt.show()

# EXEMPLO Usando simulador
simulator = qk.Aer.get_backend('qasm_simulator')
job = qk.execute(circuit, backend=simulator)
result = job.result()
plot_histogram(result.get_counts(circuit))
plt.show()

# # EXEMPLO Usando computador quântico real (é demorado)
# qk.IBMQ.load_account()
# provider = qk.IBMQ.get_provider('ibm-q')
# qcomp = provider.get_backend('ibmq_santiago')
# job = qk.execute(circuit, backend=qcomp)
# job_monitor(job)
# result = job.result()
# plot_histogram(result.get_counts(circuit))
# plt.show()