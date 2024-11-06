# https://github.com/pchung39/quantum-leap/blob/1969fea4d42e56592d0818db1d8cc70454dad44c/algorithms/simon.py
# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute

# import basic plot tools
from qiskit.visualization import plot_histogram
#from qiskit_textbook.tools import simon_oracle
import matplotlib.pyplot as plt

# going to use my own implementation of simon's oracle instead of qiskit's implementation

def simon_oracle(b):

    b = b[::-1] # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n*2)

    for qbit in range(n):
        qc.cx(qbit, qbit+n)
    
    return qc


b = '110'

n = len(b)
simon_circuit = QuantumCircuit(n*2, n)

# Apply Hadamard 
simon_circuit.h(range(n))

# Apply barrier 
simon_circuit.barrier()

simon_circuit += simon_oracle(b)
# apply Simon's oracle


# apply barrier
simon_circuit.barrier()
simon_circuit.h(range(n))

# measure qubits
simon_circuit.measure(range(n), range(n))
simon_circuit.draw(output="mpl")
plt.show()