# https://github.com/albertye1/qiskit-basics/blob/61dccbe623fb4207e1b57030d02d129d78d2156f/groveralgo/2qgrover.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram

n=2
grover_circuit = QuantumCircuit(n)

# initialize the outputs to have equal superposition
def init(qc, qubits):
	for q in range(qubits):
		qc.h(q)
	return qc

grover_circuit = init(grover_circuit, 2)
grover_circuit.draw(output = "mpl")

# apply the oracle (which changes the phases of the solution state |w> = |11>)
grover_circuit.cz(0,1)
grover_circuit.draw(output = "mpl")

# apply the diffuser
grover_circuit.h([0,1])
grover_circuit.z([0,1])
grover_circuit.cz(0,1)
grover_circuit.h([0,1])
grover_circuit.draw(output = "mpl")

""" SIMULATOR RUN """
grover_circuit.measure_all()
aer_sim = Aer.get_backend('aer_simulator')
qobj = assemble(grover_circuit)
result = aer_sim.run(qobj).result()
counts = result.get_counts()
plot_histogram(counts, color="midnightblue")

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
plot_histogram(answer)
plt.show()