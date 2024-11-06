# https://github.com/esfinkel/quantum_artist/blob/cd40106bc7fb555667d21c6beed3be40176d213c/simple_quantum_rng.py
# with thanks to Russell Huffman state of https://medium.com/qiskit/rothko-inspired-generative-quantum-art-6f34ca9d17cb
# for providing the code for this part of the project

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()

circuit = QuantumCircuit(2, 2)

circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0,1], [0,1])

# circuit.draw()

# pick a device to run on
backend = provider.get_backend('ibmq_qasm_simulator')

# Execute the circuit on the backend
job = execute(circuit, backend, shots=1000, memory=True)

# Grab results from the job
result = job.result()

#get individual shots
memory = result.get_memory()

# need an array to drop all the results into
outputArray = []

#convert results to int and drop into array
for x in range(0, 1000):
    outputArray.append(memory[x])
    
# print(outputArray)
print(outputArray.count('00'))
print(outputArray.count('11'))
print(outputArray.count('01'))
print(outputArray.count('10'))