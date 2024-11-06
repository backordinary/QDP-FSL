# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/one_qubit_gates/y_gate.py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_bloch_multivector

sim = Aer.get_backend('aer_simulator')

# Pauli gates
# qubit in state |0>
qc_original = QuantumCircuit(1)

# Apply x-gate
qc_original.save_statevector()
qobj_original = assemble(qc_original)
state = sim.run(qobj_original).result().get_statevector()
plot_bloch_multivector(state, title="original")

qc_y = QuantumCircuit(1)
qc_y.y(0)
qc_y.draw()

qc_y.save_statevector()
qobj_y = assemble(qc_y)
state = sim.run(qobj_y).result().get_statevector()
plot_bloch_multivector(state, title="Y gate applied")
plt.show()
## Note the bloch sphere only shows the relative shares of the two base vectors |0>, |1> phases like -1, i, -i are
## not depicted
