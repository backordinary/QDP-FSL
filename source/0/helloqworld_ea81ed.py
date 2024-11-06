# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/helloQWorld.py
import getpass, time
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import available_backends, execute, register, least_busy, get_backend
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram, circuit_drawer

q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)
qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)
register(Qconfig.APItoken)
backend = get_backend('ibmqx4')
job_exp = execute(qc, backend=backend, shots=1024, max_credits=3)

lapse = 0
interval = 30
while not job_exp.done:
    print('Status @ {} seconds'.format(interval * lapse))
    print(job_exp.status)
    time.sleep(interval)
    lapse += 1
print(job_exp.status)
plot_histogram(job_exp.result().get_counts(qc))
print('You have made entanglement!')