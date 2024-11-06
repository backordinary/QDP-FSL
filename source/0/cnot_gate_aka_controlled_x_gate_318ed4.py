# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/multiple_qubits_gates/cnot_gate_aka_controlled_x_gate.py
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere

qc = QuantumCircuit(2)
# cnot gate. first index is control qubit, second one is target qubit
qc.cx(0, 1)
qc.draw()
svsim = Aer.get_backend('aer_simulator')
qc.save_unitary()
qobj = assemble(qc)
cnot_gate = svsim.run(qobj).result().get_unitary()
print(cnot_gate)

# Apply cnot to superposition to get entangled state
# NOTE in bra-ket notation the rightmost qubit is the first qubit !!!
qc = QuantumCircuit(2)
first_qubit = 0

qc.h(first_qubit)
qc.draw()
# svsim = Aer.get_backend('aer_simulator')
# qc.save_statevector()
# qobj = assemble(qc)
# final_state = svsim.run(qobj).result().get_statevector()
# Productstate of |0> x H|0>
# print(final_state)

# Apply C-not gate
qc.cx(0, 1)
svsim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = assemble(qc)
result = svsim.run(qobj).result()
final_state = svsim.run(qobj).result().get_statevector()
# Productstate is entangled state: First Bell state
print(final_state)
plot_histogram(result.get_counts())
# Because of the entanglement of this state, there is no single qubit measurement basis for which a specific measurement
# is guaranteed
plot_bloch_multivector(final_state)
plot_state_qsphere(final_state)
qc.draw()

plt.show()
