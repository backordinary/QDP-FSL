# https://github.com/sybarits/quantum-computing/blob/6d5c872ed64cc969aa5611e3a291811561f6415c/grovers-search-algorithm/grover.py
with open('3sat.dimacs', 'r', encoding='utf8') as f:
    dimacs = f.read()
print(dimacs)  # let's check the file is as promised

from qiskit.circuit.library import PhaseOracle
oracle = PhaseOracle.from_dimacs_file('3sat.dimacs')
print(oracle.draw())

from qiskit import QuantumCircuit
init = QuantumCircuit(3)
init.h([0,1,2])
print(init.draw())

# steps 2 & 3 of Grover's algorithm
from qiskit.circuit.library import GroverOperator
grover_operator = GroverOperator(oracle)

qc = init.compose(grover_operator)
qc.measure_all()
print(qc.draw())

# Simulate the circuit
from qiskit import Aer, transpile
sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, sim)
counts = sim.run(t_qc).result().get_counts()
print(counts)
# plot the results
from qiskit.visualization import plot_histogram
plot_histogram(counts)


