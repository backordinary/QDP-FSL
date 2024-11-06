# https://github.com/iQuHACK/2021_Airier-Lei/blob/6c1b7e2ec97ea50b5ca97c7bf399f4c3846b8d43/bell.py
from qiskit.providers.jobstatus import JobStatus
from qiskit.tools.visualization import plot_histogram
from qiskit import *
# from qiskit_ionq_provider import IonQProvider 
from time import sleep

# provider = IonQProvider()
# provider = IBMQ.get_provider('ibm-q')
# backend = provider.get_backend("ionq_qpu")
backend = Aer.get_backend('qasm_simulator')

# def cat(qc, x):
#     n = len(x)
#     qc.h(x[0])
#     for i in range(1, n):
#         qc.cx(0, x[i])

def cat(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc.to_instruction()

n = 2
qc = QuantumCircuit(n, n)
qc.append(cat(n), range(n))
qc.h(0)
qc.cx(0, 1)

qc.measure(range(n), range(n))
print(qc)

job = execute(qc, backend)

while job.status() is not JobStatus.DONE:
    print("Job status:", job.status()) 
    sleep(5)

result = job.result()
print(result.get_counts())
plot_histogram(result.get_counts(qc))
