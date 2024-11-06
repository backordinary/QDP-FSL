# https://github.com/1chooo/Quantum-Oracle/blob/2a2f8395c17129341c485d5345aa63c2be474f7c/prog02.py
""" 
Constant: Output equal 1. 
"""


""" Create the quantum Circuit. """

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

qrx = QuantumRegister(1, 'q0')
qry = QuantumRegister(1, 'q1')
cr = ClassicalRegister(1, 'c')

qc = QuantumCircuit(qrx, qry, cr)
qc.h(qrx)
qc.x(qry)
qc.h(qry)
qc.barrier()
qc.x(qry)
qc.barrier()
qc.h(qrx)
qc.h(qry)
qc.measure(qrx, cr)
qc.draw("mpl")


""" Proof through histogram. """

from qiskit import execute
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()
job = execute(qc, backend = sim, shots = 1000)
result = job.result()
counts = result.get_counts(qc)
print("Counts: ", counts)
plot_histogram(counts)


""" Also in Quantum Computer. """

from qiskit import QuantumCircuit, IBMQ, execute 
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

IBMQ.save_account('your_token')
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
qcomp = provider.get_backend('ibmq_manila')
job = execute(qc, backend = qcomp, shots = 1000)
job_monitor(job)
result = job.result()
counts = result.get_counts(qc)
print("Total counts for qubit states are:", counts)
plot_histogram(counts)