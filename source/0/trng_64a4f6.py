# https://github.com/kishan-shanthakumar/Projects/blob/82db9fa2d710772fb9f8849e086f9da44efc1e82/Python/Quantum%20Computing/TRNG.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, IBMQ
IBMQ.load_accounts()
from qiskit.tools.visualization import plot_histogram
from qiskit import execute
import warnings
from math import pi
warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
n = 3
q = QuantumRegister(n)
c = ClassicalRegister(n)
circ = QuantumCircuit(q , c)
'''
for j in range(n):
    circ.h(q[j])
    
circ.measure(q,c)

job = execute(circ, backend = 'qasm_simulator', shots = 70000)

# get the histogram of bit string results, convert it to integer
bit_counts = job.result().get_counts()
int_counts = {}
for bitstring in bit_counts:
    int_counts[ int(bitstring,2) ] = bit_counts[bitstring]


plot_histogram(int_counts)
'''
for j in range(n):
    circ.rx(pi/(2.5+j),q[j])
    
circ.measure(q,c)
    
job = execute(circ, backend =  'qasm_simulator', shots=7000)

bit_counts = job.result().get_counts()
int_counts = {}
for bitstring in bit_counts:
    int_counts[ int(bitstring,2) ] = bit_counts[bitstring]
plot_histogram(int_counts)


