# https://github.com/jezerjojo14/QuantumSawtoothMap/blob/9e5d2de47d0efb6e9345aa48a15b773cd395713b/qSawtoothMap.py
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit.library import QFT
from math import pi
from qiskit.visualization import plot_histogram


# Function copied from qiskit textbook that implements a qft without the swaps
# Has been tweaked such that the inverse QFT would have had the SWAP gates to the right
# Alternatively, we could have avoided making changes but instead of using QFT to go from position to momentum basis,
# we'd have to pick a convention where QFT_inverse goes from position to momentum basis. I think this is what's done in the paper.

def qft_rotations(circuit, n, i=0):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if i == n:
        return circuit
    circuit.h(i)
    for qubit in range(n-1-i):
        qubit+=i+1
        circuit.cp(pi/2**(qubit-i), qubit, i)
    # At the end of our function, we call the same function again having incremented i
    i += 1
    qft_rotations(circuit, n,i)


# Input parameters

n=int(input("Enter n (number of qubits): "))
k=float(input("Enter k value: "))
T=float(input("Enter T value: "))
t=int(input("Enter t (number of iterations): "))
N=2**n


qc=QuantumCircuit(n)


# Create QFT circuit for later use

qft=QuantumCircuit(n)
qft_rotations(qft, n)


# Initial state is m = 0

qc.x(n-1)

for iter in range(t):
    # We change to position/angle basis

    qc.compose(qft.inverse(), inplace=True)

    qc.barrier()

    # Uk implementation

    # The q0 should correspond to the smallest value while q(n-1) should represent the largest
    # So for q0, j will be n. For q_i, j=n-i

    # Furthermore for Uk specifically, we count the qubits backwards because it's sandwiched between QFT_inverse and QFT.
    # Our QFT circuit is optimized such that the qubits in Uk have to be swapped in order to get proper results.

    for i in range(n):
        qc.p(k*2*(pi**2)*((2**(2*(i-n))) - (2**(i-n))) , n-1-i)
        for l in range(n-1-i):
            l+=i+1
            qc.cp(k*4*(pi**2)*((2**(i + l - 2*n))), n-1-i, n-1-l)


    qc.barrier()

    # Change back to momentum basis

    qc.compose(qft, inplace=True)

    qc.barrier()


    # Ut implementation

    for i in range(n):
        qc.p(-T*((N**2)/2)*((2**(2*(i-n))) - (2**(i-n))) , i)
        for j in range(n-1-i):
            j+=i+1
            qc.cp(-T*(N**2)*((2**(i + j - 2*n))) , i,j)


    qc.barrier()

qc.measure_all()

# Print circuit

print(qc.draw(output="text"))


# Get counts

simulator = Aer.get_backend('qasm_simulator')
result = simulator.run(transpile(qc, simulator)).result()
counts = result.get_counts(qc)

print(counts)

fig=plot_histogram(counts)
fig.savefig("t"+str(t))
