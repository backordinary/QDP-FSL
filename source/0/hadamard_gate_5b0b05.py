# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/one_qubit_gates/hadamard_gate.py
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

qc_h = QuantumCircuit(1)
qc_h.h(0)
qc_h.draw(output='mpl', filename="hadamard_gate_applied_to_0.png")

qc_h.save_statevector()
qobj_h = assemble(qc_h)
state = sim.run(qobj_h).result().get_statevector()
plot_bloch_multivector(state, title="Hadamard gate applied")
plt.show()
## Note the bloch sphere only shows the relative shares of the two base vectors |0>, |1> phases like -1, i, -i are
## not depicted
