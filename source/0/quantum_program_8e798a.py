# https://github.com/VatsKan/quantum-qosf-application/blob/092be8e2142a9143273cabe9a15cc4e911bcbfea/quantum_program.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

arr = [1,5,7,10]

#num of bits needed for storing a single integer is floor(log_2(max_int)) where max_int is largest int in the array

qc = QuantumCircuit(6) #2 address qubits, and 4 qubits for memory.
#address qubits |00> |01> |10> |11> (represent indexes)
#bin reps: 0001, 0101, 0111, 1010 

# TO FIGURE OUT:
#how to convert integers to binary rep using quantum circuit? or is this part done classically?
#how to store reps in a single superpositon qubit state using bucket-bridget method in qram article? 
#I am guessing superposition state is create by applying hadamard gates.
#how to find which indicis have adjacent bits with a quantum circuit?...probably using Grover's algorithm
#which gates to use to make the amplitude of qubits with non-adjacent bits zero in the superposition state?

qc.x(1)

qc.draw(output='mpl', filename='circuit.png') 

# sim = AerSimulator()
# job = sim.run(qc)
# counts = job.result().get_counts()
# print(counts)

