# https://github.com/keshavbharadwaj/Quantum-circuits-in-python/blob/0c620f78180e6515554855ae333de496f9670530/Deutsch%20Joza%20Algorithm/djalgorithm.py
import qiskit as q
from qiskit.visualization import plot_histogram,plot_bloch_multivector
import matplotlib
import math
qasm_sim=q.Aer.get_backend("qasm_simulator")
statevec_sim=q.Aer.get_backend("statevector_simulator")
def blackbox_constant(c):#implements a constant function
    return c
def blackbox_balanced(c):#implements a balanced function
    n_qubits=circuit.num_qubits-1
    for i in range(n_qubits):
        c.cx(i,n_qubits)
    return c

def hadamard(c):#This function is used to apply hadamard to n qbiits
    n_qubits=circuit.num_qubits
    for i in range(n_qubits-1):
        c.h(i)
    return c
n=int(input("Enter the number of qbits : "))# number of input bits
y=int(input("Choose 1 if you want balanced or 0 if you want constant function"))#function selection
circuit=q.QuantumCircuit(n+1,n+1)#quantumbits = n+1 and classicalbits=n+1
circuit.x(n)
circuit.barrier()#Used for better visualization
circuit=hadamard(circuit)
circuit.h(n)
circuit.barrier()
if y==1:
    circuit=blackbox_balanced(circuit)#calling the balanced function
if y==0:
    circuit=blackbox_constant(circuit)#calling the constant function
circuit.barrier()
circuit=hadamard(circuit)
circuit.draw(output='mpl',filename='djckt.png')
originalstate=q.execute(circuit,backend=statevec_sim).result().get_statevector()
circuit.measure([i for i in range(n)],[i for i in range(n)])#measuring n quantum bits to n classical bits
original_count=q.execute(circuit,backend=qasm_sim).result().get_counts()
print(circuit)
plot_bloch_multivector(originalstate).show()
plot_histogram(original_count).show()

