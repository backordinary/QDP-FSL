# https://github.com/ntanetani/FRQI/blob/2c85fd4f78ecd0d0c14ff38349c35e583323511a/src/frqi_real.py
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
import numpy as np

IBMQ.load_account()
#provider = IBMQ.get_provider(hub='ibm-q-keio', group='keio-internal', project='keio-students')
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q,c)


def rry(circ, con1, con2, angle, t, c0, c1):

    circ.h(t)

    circ.rz(angle/4, t)

    if(con2 == 0):
        circ.x(c0)

    circ.cx(c0, t)

    if(con2 == 0):
        circ.x(c0)

    circ.rz(-angle/4, t)

    if(con1 == 0):
        circ.x(c1)

    circ.cx(c1, t)

    if(con1 == 0):
        circ.x(c1)

    circ.rz(angle/4, t)

    if(con2 == 0):
        circ.x(c0)

    circ.cx(c0, t)

    if(con2 == 0):
        circ.x(c0)

    circ.rz(-angle/4, t)

    circ.h(t)

for i in range(1,len(q)):
    qc.h(q[i])

rry(qc, 0, 0, 2 * np.pi/2, q[0], q[1], q[2])
rry(qc, 0, 1, 2 * np.pi/4, q[0], q[1], q[2])
rry(qc, 1, 0, 2 * np.pi/6, q[0], q[1], q[2])
rry(qc, 1, 1, 2 * np.pi/8, q[0], q[1], q[2])

for i in range(len(q)):
    qc.measure(q[i],c[i])

# backends = ['ibmq_20_tokyo', 'ibmq_qasm_simulator']

# Use this for the real machine
backend_sim = provider.get_backend('ibmq_london')

# Use this for the simulator
#backend_sim = provider.get_backend('ibmq_qasm_simulator')

result = execute(qc, backend_sim, shots=4096).result()
#circuit_drawer(qc).show()
plot_histogram(result.get_counts(qc))

print(result.get_counts(qc))
