# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/w_states/test_w3.py
# from qiskit import Quan  # useful additional packages
# import matplotlib.pyplot as plt
import numpy as np

# from pprint import pprinttumCircuit
from qiskit import execute, Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, IBMQ
from qiskit.tools.visualization import circuit_drawer, plot_histogram
from qiskit.backends.ibmq import least_busy


# Here, two useful routine
# Define a F_gate
def F_gate(circ, q, i, j, n, k):
    theta = np.arccos(np.sqrt(1 / (n - k + 1)))
    circ.ry(-theta, q[j])
    circ.cz(q[i], q[j])
    circ.ry(theta, q[j])
    circ.barrier(q[i])


# Define the cxrv gate which uses reverse CNOT instead of CNOT
def cxrv(circ, q, i, j):
    circ.h(q[i])
    circ.h(q[j])
    circ.cx(q[j], q[i])
    circ.h(q[i])
    circ.h(q[j])
    circ.barrier(q[i], q[j])


# True random real device, False simulator, None ibmqx2
# Just a quick hack
real = True
n = 3
q = QuantumRegister(n)
c = ClassicalRegister(n)

qc = QuantumCircuit(q, c)
qc.x(q[2])  #start is |100>
F_gate(qc, q, 2, 1, 3, 1)  # Applying F12
F_gate(qc, q, 1, 0, 3, 2)  # Applying F23
# flag_qx2 = True
# if backend.name() == 'ibmqx4':
#         flag_qx2 = False
# if flag_qx2:  # option ibmqx2
#     qc.cx(q[1], q[2])  # cNOT 21
#     qc.cx(q[0], q[1])  # cNOT 32
# else:  # option ibmqx4, we simulate a cx w/ different controls and target
#     cxrv(qc, q, 1, 2)
#     cxrv(qc, q, 0, 1)

qc.cx(q[1], q[2])  # cNOT 21
qc.cx(q[0], q[1])  # cNOT 32
qc.measure(q, c)

circuit_drawer(qc, filename="imgs/w3.png")
shots = 4098
if real is None:
    IBMQ.load_accounts()
    backend = IBMQ.get_backend('ibmqx2')
elif real:
    # using a real device
    IBMQ.load_accounts()
    large_enough_devices = IBMQ.backends(
        filters=
        lambda x: x.configuration()['n_qubits'] >= n and not x.configuration()['simulator']
    )
    backend = least_busy(large_enough_devices)
else:
    # using local qasm simulator
    backend = Aer.get_backend('qasm_simulator')

print("Executing on", backend)
job = execute(qc, backend=backend, shots=shots)
jobID = job.job_id()
print("Your job id is " + jobID)
counts = job.result().get_counts(qc)
print(counts)
plot_histogram(counts)
