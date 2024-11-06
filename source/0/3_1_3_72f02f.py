# https://github.com/dsaf2007/QuantumComputing/blob/08e3dcc6f9210f4605e6eb1996b3534e43326141/3_1_3.py
import sys,time
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit import IBMQ

# Generate an 2**n bit random number where n = # of qubits
def qrng(n):
 # create n qubit(s)
 quantum_r = QuantumRegister(n)
 # create n classical registers
 classical_r = ClassicalRegister(n)
 # create a circuit
 circuit = QuantumCircuit(quantum_r,classical_r)
 # enable logging

 # Hadamard gate to all qubits
 for i in range(n):
    circuit.h(quantum_r[i])
 # measure qubit n and store in classical n
 for i in range(n):
     circuit.measure(quantum_r[i], classical_r[i])
 # backend simulator
 backend = Aer.get_backend('qasm_simulator')
 shots=1024
 #execute circuit on qasm simulator
 job_sim = backend.run(transpile(circuit,backend),shots)
 #grab results from job
 result_sim = job_sim.result()
# Show result counts
 # counts={'100': 133, '101': 134, '011': 131, '110': 125, '001': 109, '111': 128, '010': 138, '000': 126}
 counts = result_sim.get_counts(circuit)
 print(counts)
 
 bits = ""
 for v in counts.values():
    if v > shots/(2**n) :
        bits += "1"
    else:
        bits += "0"
 return int(bits, 2)
###########################################
if __name__ == '__main__':
 start_time = time.time()
 numbers = []
 # generate 100 8 bit rands
 size = 10
 qubits = 3 # bits = 2**qubits
 for i in range(size):
    n = qrng(qubits)
    numbers.append(n)
 print ("list=" + str(numbers))
 print("--- %s seconds ---" % (time.time() - start_time))

