# https://github.com/G-Carneiro/GCQ/blob/a557c193c54c4f193c9ffde7f94c576b06972abe/src/teletransporte.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.ignis.verification import marginal_counts
from qiskit.providers.aer.backends.aer_simulator import AerSimulator

psi = QuantumRegister(1, name="psi")
bell = QuantumRegister(2, name="bell")
bell_0 = bell.__getitem__(0)
bell_1 = bell.__getitem__(1)
c_0 = ClassicalRegister(1)
c_1 = ClassicalRegister(1)
c_result = ClassicalRegister(1)
quantum_circuit: QuantumCircuit = QuantumCircuit(psi, bell, c_0, c_1, c_result)

# create bell 00 state
quantum_circuit.h(1)
quantum_circuit.cx(bell_0, bell_1)

# setup alice qubit
quantum_circuit.cx(psi, bell_0)
quantum_circuit.h(psi)
quantum_circuit.measure(psi, c_0)
quantum_circuit.measure(bell_0, c_1)

# setup bob
quantum_circuit.x(bell_1).c_if(c_1, 1)
quantum_circuit.z(bell_1).c_if(c_0, 1)
quantum_circuit.measure(bell_1, c_result)

aer_sim = AerSimulator()
counts = aer_sim.run(quantum_circuit).result().get_counts()
qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3)]

print(quantum_circuit)
print(qubit_counts)
