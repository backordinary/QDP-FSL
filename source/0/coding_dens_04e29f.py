# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/coding_dens.py
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

qc = QuantumCircuit(2,2)

# Bell Gate
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0,1)
bell.to_gate()
bell.name = 'Bell state'

# Measure Bell

measure = QuantumCircuit(2)
measure.cx(0,1)
measure.h(0)
measure.to_gate()
measure.name = 'Measure in Bell state'

qc.append(bell,[0,1])
qc.append(measure,[0,1])
qc.measure([0,1], [0,1])
bell.draw(output='mpl')
measure.draw(output='mpl')
qc.draw(output='mpl')

simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, backend = simulator).result()
plot_histogram(result.get_counts(qc))
plt.show()

IBMQ.load_account()
host = IBMQ.get_provider('ibm-q')
quantum_computer = host.get_backend('ibmq_santiago')
result_qcomputer = execute(qc, backend= quantum_computer)

job_monitor(result_qcomputer)
result_qcomputer = result_qcomputer.result()
plot_histogram(result_qcomputer)
plt.show()

