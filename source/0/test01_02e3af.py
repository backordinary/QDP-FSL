# https://github.com/DCYokaze/ibmq-code/blob/765bf3b309fca0cb03c4b75795865a0b2a04cecc/test01.py
# Click 'try', then 'run' to see the output,
# you can change the code and run it again.
print("This code works!")
from qiskit import QuantumCircuit
qc = QuantumCircuit(2) # Create circuit with 2 qubits
qc.h(0)    # Do H-gate on q0
qc.cx(0,1) # Do CNOT on q1 controlled by q0
qc.measure_all()
print(qc)
# qc.draw(output='text')
# qc.draw(output='latex')
qc.draw(output='mpl')
