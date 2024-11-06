# https://github.com/AlbertoVari/SolidQML/blob/fbe3c66483c4d82b2d945151733bb9d157dcc237/accessIBMQ.py
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
IBMQ.load_account()
provider = IBMQ.load_account()
#Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
# Add a H gate on qubit 0
circuit.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)
# Grab results from the job
result = job.result()
# Return counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# END
