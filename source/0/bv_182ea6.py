# https://github.com/albertye1/qiskit-basics/blob/a7b7a5c85e5174592adc10c150ed02059cdf598b/bernstein-vazirani/bv.py
# kinda similar to deutsch-josza but we are looking for something else
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.visualization import plot_histogram

n = 3
s = '101'
bv_circuit = QuantumCircuit(n+1, n) # n-qubit output measurement
bv_circuit.h(n)
bv_circuit.z(n)
for i in range(n):
	bv_circuit.h(i) # equal superposition
bv_circuit.barrier()

s = s[::-1] # reverse the string s
for q in range(n):
	if s[q] == '0':
		bv_circuit.i(q) # nothing
	else:
		bv_circuit.cx(q,n)
bv_circuit.barrier()

for i in range(n):
	bv_circuit.h(i)
for i in range(n):
	bv_circuit.measure(i, i)
bv_circuit.draw(output='mpl')
plt.show()

# use local simulator
aer_sim = Aer.get_backend('aer_simulator')
shots = 1024
qobj = assemble(bv_circuit)
results = aer_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()
# exit(0)

# Load our saved IBMQ accounts and get the least busy backend device with less than or equal to 5 qubits
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits <= 5 and
                                   x.configuration().n_qubits >= 2 and
                                   not x.configuration().simulator and x.status().operational==True))
print("least busy backend: ", backend)

# Run our circuit on the least busy backend. Monitor the execution of the job in the queue
from qiskit.tools.monitor import job_monitor

shots = 1024
transpiled_bv_circuit = transpile(bv_circuit, backend)
job = backend.run(transpiled_bv_circuit, shots=shots)

job_monitor(job, interval=2)

# Get the results from the computation
results = job.result()
answer = results.get_counts()

plot_histogram(answer)
plt.show()