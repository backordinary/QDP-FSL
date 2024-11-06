# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw3/xor.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt

# Quantum XOR - really just CNOT
# input qubits: a, b
# output qubits: b 
def qxor(circ, a, b):
    circ.cx(a, b)


# Create registers and circuit
a = QuantumRegister(2, 'a')  # input
qc = QuantumCircuit(a)

# Build a circuit that initializes the input bits to a full superposition
# Calls the XOR function and then measures the output register

qc.h(a) # puts hadamard on both qubits in register
# barrier will prevent transpile (called by execute) to restrict optimizations between barriers 
# and not entire circuit, won't combine gates and make optimizations across barriers
qc.barrier()
qxor(qc, a[0], a[1])
qc.barrier()
m = ClassicalRegister(1, 'output')
qc.add_register(m)
qc.measure(a[1], m)

# Simulate and show results
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=4096)  # shots default = 1024
result = job.result()
print(result.get_counts())

qc.draw(output='mpl')
plot_histogram(result.get_counts()) # output is just the second qubit activated/inactivated by CNOT
plt.show()