# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/grovers_boolean.py
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PhaseOracle
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
# Criando o circuito grover #

def reflection(n):
    qc = QuantumCircuit(n)
    qc.h(list(range(0,n)))
    qc.x(list(range(0,n)))
    # A porta Multi-controladora-Z
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1)  # Multi-controladora toff
    qc.h(n-1)
    qc.x(list(range(0,n)))
    qc.h(list(range(0,n)))
    return qc

# Porta Oraculo
n = 4
expr = '(a & ~b & ~c) | (~a & ~b & d) | ~(a | d & c) & (a|~d |~c) | (a & b & ~c)'
oracle = PhaseOracle(expr)
oracle.to_gate()

# Porta de Grover
g = reflection(n)
g.to_gate()
g.name = 'G'

backend = Aer.get_backend('qasm_simulator')
grover_circuit = QuantumCircuit(n,n)
grover_circuit.h(list(range(0,n)))
grover_circuit.append(oracle, list(range(0,n)))
grover_circuit.append(g, list(range(0,n)))
grover_circuit.measure(list(range(0,n)),list(range(0,n)))
grover_circuit.draw(output='mpl')
oracle.draw(output='mpl')
g.draw(output='mpl')

job = execute(grover_circuit, backend, shots = 1024)
result = job.result()
counts = result.get_counts()
plot_histogram(counts)
plt.show()

IBMQ.load_account()
host = IBMQ.get_provider('ibm-q')
quantum_computer = host.get_backend('ibmq_lima')
result_qcomputer = execute(grover_circuit, backend= quantum_computer)

job_monitor(result_qcomputer)
result_qcomputer = result_qcomputer.result()
plot_histogram([result_qcomputer.get_counts(grover_circuit), counts], legend=["Qcumputer","simulated"], bar_labels=True)
plt.ylabel("probabilidades")
plt.xlabel("Possíveis Combinações")
plt.show()