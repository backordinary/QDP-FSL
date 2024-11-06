# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/entanglement.py
from time import sleep

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.backends.jobstatus import JOB_FINAL_STATES
from qiskit.tools.visualization import plot_histogram, circuit_drawer, matplotlib_circuit_drawer

q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

# entanglement:
qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

qiskit.IBMQ.load_accounts()
backend = qiskit.IBMQ.get_backend('ibmq_qasm_simulator')

job_exp = execute(qc, backend=backend, shots=1024, max_credits=3)
print("Job ID %s" % job_exp.job_id())

while job_exp.status() not in JOB_FINAL_STATES:
    print(job_exp.status())
    sleep(1)
print(job_exp.status())

matplotlib_circuit_drawer(qc)
plot_histogram(job_exp.result().get_counts(qc))
print('We made entanglement of two qubits!')

