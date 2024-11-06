# https://github.com/calwoo/qlearn/blob/8ee23033baf69091c944132378f718ea6c4ce950/qubits/adder.py
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram


n = 8
n_q = n  # num qubits
n_b = n  # num output bits
qc_output = QuantumCircuit(n_q, n_b)
for j in range(n):
    qc_output.measure(j, j)

qc_encode = QuantumCircuit(n)
qc_encode.x(7)

qc = qc_encode + qc_output

# simulate results
sim = Aer.get_backend("qasm_simulator")
qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)
plt.show()

### half-adder circuit
qc_ha = QuantumCircuit(4, 2)
qc_ha.x(0)
qc_ha.x(1)
qc_ha.barrier()
qc_ha.cx(0, 2)
qc_ha.cx(1, 2)
# toffoli gate for AND
qc_ha.ccx(0, 1, 3)
qc_ha.barrier()

qc_ha.measure(2, 0)
qc_ha.measure(3, 1)

qc_ha.draw(output="mpl")
plt.show()

qobj = assemble(qc_ha)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)
plt.show()
