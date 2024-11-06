# https://github.com/4-space/Quantum-Nearest-Neighbors/blob/e505f882a5a00c924221f34b4c925933b7f7c962/quantum_nearest_neighbors.py

from qiskit import QuantumProgram, QuantumCircuit
from qiskit.extensions.standard import header
from qiskit import CompositeGate

API_FILE = open("API_TOKEN.txt")
API_TOKEN = API_FILE.readlines()[0]

from qiskit.tools.visualization import plot_histogram

qp = QuantumProgram()
qp.set_api(API_TOKEN, API_URL)

q = qp.create_quantum_register('q', 5)
c = qp.create_classical_register('c', 1)
cl = qp.create_circuit('cl', [q], [c])

#prepare states

cl.h(q[0])
cl.h(q[1])
cl.barrier(q)

cl.cu3(4.304, 0, 0, q[0], q[1])

cl.x(q[0])

cl.ccx(q[0], q[1], q[2])

cl.x(q[1])

#begin controlled-controlled u3
cl.barrier(q)

cl.cx(q[1], q[2])
cl.u3(-0.331, 0, 0, q[2])
cl.ccx(q[0], q[1], q[2])
cl.cx(q[1], q[2])
cl.u3(-0.331, 0, 0, q[2])
cl.cx(q[1], q[2])
cl.u3(0.331, 0, 0, q[2])

#end controlled-controlled u3
cl.barrier(q)

cl.swap(q[2],q[3])
cl.cx(q[2],q[1])
cl.h(q[0])

cl.barrier(q)
#measure qbits
cl.measure(q[2],c[0])


#running

backend_test = "local_qasm_simulator"
shots = 8000

result = qp.execute('cl', backend=backend_test, timeout=2400, shots=shots)


data = result.get_counts('cl')

plot_histogram(data)
