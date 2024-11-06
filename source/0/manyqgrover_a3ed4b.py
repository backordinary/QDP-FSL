# https://github.com/albertye1/qiskit-basics/blob/5a31c03601ba2d92de892660cd6e1805c1e31aaa/groveralgo/manyqgrover.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram

# can go very high but it's expo time so it'll be very slow too
n=6
grover_circuit = QuantumCircuit(n)

# initialize the outputs to have equal superposition
def init(qc, qubits):
	for q in range(qubits):
		qc.h(q)
	return qc

grover_circuit = init(grover_circuit, n)
grover_circuit.draw(output = "mpl")

oc = QuantumCircuit(n)
oc.cz(0, 5)
oc.cz(1, 5)
oc.cz(3, 5)
oc.name = "U$_\omega$"
oracle = oc.to_gate()
# apply the oracle (which changes the phases of the solution state |w> = |11>)
grover_circuit.append(oracle, [0,1,2,3,4,5])
grover_circuit.draw(output = "mpl")

# apply the diffuser
def diff(n):
	qc = QuantumCircuit(n)
	# H everyone
	for q in range(n):
		qc.h(q)
	# X everyone
	for q in range(n):
		qc.x(q)
	# multi-controlled Z gate
	qc.h(n-1)
	qc.mct(list(range(n-1)), n-1)
	qc.h(n-1)
	# X everyone
	for q in range(n):
		qc.x(q)
	# H everyone
	for q in range(n):
		qc.h(q)
	diffuser = qc.to_gate()
	diffuser.name = "U$_s$"
	return diffuser

grover_circuit.append(diff(n), [0,1,2,3,4,5])
grover_circuit.measure_all()
grover_circuit.draw(output = "mpl")
plt.show()

""" SIMULATOR RUN """
aer_sim = Aer.get_backend('aer_simulator')
transpiled_grover_circuit = transpile(grover_circuit, aer_sim)
qobj = assemble(transpiled_grover_circuit)
results = aer_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts, color="red")

""" REAL QUANTUM COMPUTER (REAL) """
provider = IBMQ.load_account()
provider = IBMQ.get_provider("ibm-q")
device = least_busy(provider.backends(filters=lambda x: int(x.configuration().n_qubits) >= 3 and 
                                   not x.configuration().simulator and x.status().operational==True))
print("Running on current least busy device: ", device)

from qiskit.tools.monitor import job_monitor
transpiled_grover_circuit = transpile(grover_circuit, device, optimization_level=3)
job = device.run(transpiled_grover_circuit)
job_monitor(job, interval=2)

# Get the results from the computation
results = job.result()
answer = results.get_counts(grover_circuit)
plot_histogram(answer, color="green")
plt.show()