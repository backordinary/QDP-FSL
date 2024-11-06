# https://github.com/parasol4791/quantumComp/blob/101be82cc0cf0baa617cae947484534eb71e22fd/visualize.py
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from math import sqrt

sim = Aer.get_backend('qasm_simulator')

# Plotting count histogram and Bloch sphere (default for state 0)
qc = QuantumCircuit(1)
qc.x(0)
qc.h(0)
qc.save_statevector()
result = sim.run(qc).result()
counts = result.get_counts()
print(counts)
# Interactive - from debugger
plot_histogram(counts, color='midnightblue', title="Qubit state counts")

state = result.get_statevector()
print(state)
# Interactive - from debugger
plot_bloch_multivector(state)

# Generates Latex object. Not sure what to do with it in Python
l = array_to_latex(state, prefix="\\text{Statevector} = ")
print(l.data)


# Visualizing entangled state
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.save_statevector()
qc.measure_all()
result = sim.run(qc).result()
counts = result.get_counts()
print(counts)
# Interactive - from debugger
plot_histogram(counts, color='midnightblue', title="Qubit state counts")

state = result.get_statevector()
print(state)
# Interactive - from debugger
plot_bloch_multivector(state)


qc = QuantumCircuit(2)
qc.x(1)
qc.save_unitary()  # saves matrix
unitary = sim.run(qc).result().get_unitary()
print(unitary)
u = array_to_latex(unitary, source=True, prefix="\\text{Circuit = } ")
print(u)
#  0 & 0 & 1 & 0  \\
#  0 & 0 & 0 & 1  \\
#  1 & 0 & 0 & 0  \\
#  0 & 1 & 0 & 0  \\
# In essence, this is matrix
# 0 I
# I 0

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
qc.save_unitary()  # saves matrix
unitary = sim.run(qc).result().get_unitary()
print(unitary)
u = array_to_latex(unitary, source=True, prefix="\\text{Circuit = } ")
print(u)
# \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}}  \\
#  \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & \tfrac{1}{\sqrt{8}} & -\tfrac{1}{\sqrt{8}}  \\
# This is matrix
# 1/sqrt(8) * ones(8,8)

