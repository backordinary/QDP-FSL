# https://github.com/Armalite/lets-try-qisbit/blob/e2e164880f12e6406c508152a250b62c5bc53f15/qisbit_test.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
# Create quantum circuit with 3 qubits and 3 classical bits
qc = QuantumCircuit(3, 3)
qc.measure([0,1,2], [0,1,2])
qc.draw()  # returns a drawing of the circuit

sim = AerSimulator()  # make new simulator object

job = sim.run(qc)      # run the experiment
result = job.result()  # get the results
print(result.get_counts())    # interpret the results as a "counts" dictionary

# Create quantum circuit with 3 qubits and 3 classical bits:
qc = QuantumCircuit(3, 3)
qc.x([0,1])  # Perform X-gates on qubits 0 & 1
qc.measure([0,1,2], [0,1,2])
print(qc.draw())    # returns a drawing of the circuit



# Create quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)
qc.x(0)
qc.cx(0,1)  # CNOT controlled by qubit 0 and targeting qubit 1
qc.measure([0,1], [0,1])
print(qc.draw())     # display a drawing of the circuit

job = sim.run(qc)      # run the experiment
result = job.result()  # get the results
# interpret the results as a "counts" dictionary
print("Result: ", result.get_counts())


test_qc = QuantumCircuit(4, 2)

# First, our circuit should encode an input (here '11')
test_qc.x(0)
test_qc.x(1)

# Next, it should carry out the adder circuit we created
test_qc.cx(0,2)
test_qc.cx(1,2)
test_qc.ccx(0,1,3)

# Finally, we will measure the bottom two qubits to extract the output
test_qc.measure(2,0)
test_qc.measure(3,1)
print(test_qc.draw())