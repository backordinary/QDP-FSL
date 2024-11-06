# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/one_qubit_gates/p_gate.py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

sim = Aer.get_backend('aer_simulator')

# p-gate is parametrised by angle phi.
# it performs a rotation of phi around the z-axis

qc = QuantumCircuit(1)
qc.p(pi / 4, 0)
qc.draw()

# Three typical P gates are
# 1. The I-gate (identity)
qc_i = QuantumCircuit(1)
qc_i.i(0)
qc_i.draw()

qc_p = QuantumCircuit(1)
qc_p.p(1 / 2 * pi, 0)
qc_p.draw()

# 2. The S gate or sqrt(Z) gate. It is the first gate in this folder that is not its own inverse.
# The inverse gate is called S-dagger
# qc = QuantumCircuit(1)
qc.s(0)  # Apply S-gate to qubit 0
qc.sdg(0)  # Apply Sdg-gate to qubit 0
qc.draw()
# 3. The T gate sqrt_4(z) gate. It is the another gate in this folder that is not its own inverse.
# The inverse gate is called T-dagger
qc = QuantumCircuit(1)
qc.t(0)  # Apply T-gate to qubit 0
qc.tdg(0)  # Apply Tdg-gate to qubit 0
qc.draw()

plt.show()
