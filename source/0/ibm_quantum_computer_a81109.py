# https://github.com/sujitmandal/Quantum-Programming/blob/1d348c307584bb804255eab75357789786ac5437/IBM_quantum_computer.py
import qiskit
from qiskit import IBMQ
from qiskit import execute
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram

'''
This programe is create by Sujit Mandal
Github: https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
'''

qr = QuantumRegister(2)
cr = ClassicalRegister(2)

circuit = QuantumCircuit(qr, cr)
circuit.draw(output='mpl')
plt.show()

circuit.cx(qr[0], qr[1])
circuit.draw(output='mpl')
plt.show()

circuit.measure(qr, cr)
circuit.draw(output='mpl')
plt.show()
IBMQ.save_account('') #ibm kye
IBMQ.load_account()

provider = IBMQ.get_provider('ibm-q')
quantum_computer = provider.get_backend('ibmq_16_melbourne')
job = execute(circuit, backend=quantum_computer)
job_monitor(job)

result = job.result()
plot_histogram(result.get_counts(circuit))
plt.show()
