# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/3sat.py
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit.library import PhaseOracle, GroverOperator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

#
# -------------- building oracle ----------------------------------------
#

oracle = PhaseOracle.from_dimacs_file("./3sat.dimacs")
oracle.draw(output="mpl")
plt.show()


#
# -------------- creating equal superposition for input bits ------------
#

init = QuantumCircuit(3)
init.h([0, 1, 2])
init.draw(output="mpl")
plt.show()


#
# -------------- steps 2 & 3 of Grover's algorithm ----------------------
#

grover_operator = GroverOperator(oracle)

qc = init.compose(grover_operator)
qc.measure_all()
qc.draw(output="mpl")
plt.show()


#
# -------------- circuit simulation ----------------------
#

sim = Aer.get_backend("aer_simulator")
t_qc = transpile(qc, sim)
counts = sim.run(t_qc).result().get_counts()

plot_histogram(counts)
plt.show()
