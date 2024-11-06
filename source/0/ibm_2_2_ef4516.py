# https://github.com/jongjinna/Quantum_IBM/blob/6df03de5de8fefc014c44b6e630ca0e3a6d6b657/IBM_Ex/IBM_2_2.py
from qiskit import QuantumCircuit, Aer, assemble
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

# qc = QuantumCircuit(3)
# # Apply H-gate to each qubit:
# for qubit in range(3):
#     qc.h(qubit)
# # See the circuit:
# print(qc.draw())

# # Let's see the result
# svsim = Aer.get_backend('aer_simulator')
# qc.save_statevector()
# qobj = assemble(qc)
# final_state = svsim.run(qobj).result().get_statevector()

# In Jupyter Notebooks we can display this nicely using Latex.
# If not using Jupyter Notebooks you may need to remove the 
# array_to_latex function and use print(final_state) instead.
from qiskit.visualization import array_to_latex
# array_to_latex(final_state, prefix="\\text{Statevector} = ")

# qc = QuantumCircuit(2)
# qc.h(0)
# qc.x(1)
# print(qc.draw())

# usim = Aer.get_backend('aer_simulator')
# qc.save_unitary()
# qobj = assemble(qc)
# unitary = usim.run(qobj).result().get_unitary()

# # In Jupyter Notebooks we can display this nicely using Latex.
# # If not using Jupyter Notebooks you may need to remove the 
# # array_to_latex function and use print(unitary) instead.
# from qiskit.visualization import array_to_latex
# array_to_latex(unitary, prefix="\\text{Circuit = }\n")

# qc = QuantumCircuit(2)
# qc.x(1)
# print(qc.draw())

# # Simulate the unitary
# usim = Aer.get_backend('aer_simulator')
# qc.save_unitary()
# qobj = assemble(qc)
# unitary = usim.run(qobj).result().get_unitary()
# # Display the results:
# array_to_latex(unitary, prefix="\\text{Circuit = } ")

# qc = QuantumCircuit(2)
# # Apply CNOT
# qc.cx(0,1)
# # See the circuit:
# print(qc.draw())

# qc = QuantumCircuit(2)
# # Apply H-gate to the first:
# qc.h(0)
# print(qc.draw())

# # Let's see the result:
# svsim = Aer.get_backend('aer_simulator')
# qc.save_statevector()
# qobj = assemble(qc)
# final_state = svsim.run(qobj).result().get_statevector()
# # Print the statevector neatly:
# array_to_latex(final_state, prefix="\\text{Statevector = }")

# qc = QuantumCircuit(2)
# # Apply H-gate to the first:
# qc.h(0)
# # Apply a CNOT:
# qc.cx(0,1)
# print(qc.draw())

# # Let's get the result:
# qc.save_statevector()
# qobj = assemble(qc)
# result = svsim.run(qobj).result()
# # Print the statevector neatly:
# final_state = result.get_statevector()
# array_to_latex(final_state, prefix="\\text{Statevector = }")

# plot_histogram(result.get_counts())
# plt.show()

# plot_bloch_multivector(final_state)
# plt.show()

# from qiskit.visualization import plot_state_qsphere
# plot_state_qsphere(final_state)

# plt.show()