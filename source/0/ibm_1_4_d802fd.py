# https://github.com/jongjinna/Quantum_IBM/blob/6df03de5de8fefc014c44b6e630ca0e3a6d6b657/IBM_Ex/IBM_1_4.py
# Single Qubit Gates

from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt
sim = Aer.get_backend('aer_simulator')

# # Let's do an X-gate on a |0> qubit
# qc = QuantumCircuit(1)
# qc.x(0)
# print(qc.draw())

# # Let's see the result
# qc.save_statevector()
# qobj = assemble(qc)
# state = sim.run(qobj).result().get_statevector()
# plot_bloch_multivector(state)
# plt.show()

# # Run the code in this cell to see the widget
# from qiskit_textbook.widgets import gate_demo
# print(gate_demo(gates='pauli'))

# qc.y(0) # Do Y-gate on qubit 0
# qc.z(0) # Do Z-gate on qubit 0
# print(qc.draw())

# # Run the code in this cell to see the widget
# from qiskit_textbook.widgets import gate_demo
# gate_demo(gates='pauli+h')

# # Create the X-measurement function:
# def x_measurement(qc, qubit, cbit):
#     """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
#     qc.h(qubit)
#     qc.measure(qubit, cbit)
#     return qc

# initial_state = [1/sqrt(2), -1/sqrt(2)]
# # Initialize our qubit and measure it
# qc = QuantumCircuit(1,1)
# qc.initialize(initial_state, 0)
# x_measurement(qc, 0, 0)  # measure qubit 0 to classical bit 0
# print(qc.draw())

# qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
# counts = sim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
# plot_histogram(counts)  # Display the output on measurement of state vector
# plt.show()

# # Run the code in this cell to see the widget
# from qiskit_textbook.widgets import gate_demo
# gate_demo(gates='pauli+h+p')

# qc = QuantumCircuit(1)
# qc.p(pi/4, 0)
# print(qc.draw())

qc = QuantumCircuit(1)
qc.s(0)   # Apply S-gate to qubit 0
qc.sdg(0) # Apply Sdg-gate to qubit 0
print(qc.draw())

qc = QuantumCircuit(1)
qc.t(0)   # Apply T-gate to qubit 0
qc.tdg(0) # Apply Tdg-gate to qubit 0
print(qc.draw())

# Let's have U-gate transform a |0> to |+> state
qc = QuantumCircuit(1)
qc.u(pi/2, 0, pi, 0)
print(qc.draw())

# Let's see the result
qc.save_statevector()
qobj = assemble(qc)
state = sim.run(qobj).result().get_statevector()
plot_bloch_multivector(state)
plt.show()