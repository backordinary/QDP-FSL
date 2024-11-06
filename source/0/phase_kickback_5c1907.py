# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/multiple_qubits_gates/phase_kickback.py
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
from qiskit.providers.aer.library import save_unitary


# 1. Apply Hadarmard on 1st and 2nd qubit
# 2. Apply CNot gate
# 3. Apply Hadarmard on 1st and 2nd qubit after CNot
# Kickback is where the eigenvalue added by a gate to a qubit is ‘kicked back’ into a different qubit via a controlled
# operation.
def cnot_switched_qubits() -> QuantumCircuit:
    quantum_cicuit = QuantumCircuit(2)
    first_qubit = 0
    second_qubit = 1
    quantum_cicuit.h(first_qubit)
    quantum_cicuit.h(second_qubit)
    quantum_cicuit.cx(first_qubit, second_qubit)
    quantum_cicuit.h(first_qubit)
    quantum_cicuit.h(second_qubit)
    return quantum_cicuit


# CNOT-gate wrapped in H-gates
# q0 ----H----------H---
# q1 ----H---CNOT---H---
# This changes the computational basis from |0>, |1> to |+>, |->


# qc.save_unitary()
# Let's see the result
qc = cnot_switched_qubits()
svsim = Aer.get_backend('aer_simulator')
qc.save_unitary()
# qc.save_statevector()
qobj = assemble(qc)
unitary = svsim.run(qobj).result().get_unitary()
print(unitary)
plt.show()
