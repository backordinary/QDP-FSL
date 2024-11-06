# https://github.com/RdWeirdo981/qiskit-learning/blob/ac0e50c4ee84b0aa71e86a573c55030f14dcc430/test.py
# Click 'try', then 'run' to see the output,
# you can change the code and run it again.
print("This code works!")
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt 
qc = QuantumCircuit(2) # Create circuit with 2 qubits
qc.h(0)    # Do H-gate on q0
qc.cx(0,1) # Do CNOT on q1 controlled by q0
qc.measure_all()
qc.draw(output="mpl")
plt.show()