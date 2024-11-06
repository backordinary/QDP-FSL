# https://github.com/roxas198/Quantum-Project-Group2/blob/d52afa4412b2968db2a83be331d4555120c40c59/quantumProject.py
from qiskit import QuantumCircuit, assemble, Aer, extensions, QuantumRegister
from qiskit.circuit.library import MCMTVChain
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
import math
from qiskit.circuit.library.standard_gates import XGate

#Number of qubits is 4
nQubit = 4

#Gives a permutation of a list from 0 to N-1.
L = np.random.permutation(2**nQubit)
print("List:",L)
#Randomly select index from 0 to N-1
y = np.random.randint(2**nQubit)
print("Initial index:",y)


#initialize m and l based on the Grover's search algorithm
m = 1
l = 6 / 5.0

#Loop over Grover's algorithn 1.6sqrt(N) times
for i in range(math.ceil(1.6*nQubit)):

    #Determines the matrix for the oracle
    matrix = np.zeros((2**nQubit,2**nQubit))
    
    for n in range(2**nQubit):
        if (L[n] >= L[y]):
            matrix[n][n] = 1
        else:
            matrix[n][n] = -1

    #Adds the oracle to the circuit
    oracle = extensions.UnitaryGate(matrix)
    #Initialize the circuit for Grover's Algorithm
    groverAlgorithm = QuantumCircuit(nQubit)

    #Apply Hadamard transform to all qubits. This gives the quantum superposition for ampltiude amplification
    #That is, gives the state of 1/sqrt(N) sum_{x=0}^{N-1} |x>, where N = 2^nQubit
    groverAlgorithm.h(range(nQubit))

    #Randomly generate j to be 1<= j < m
    j = np.random.randint(m)

    #Loop over Grover's search subroutine j times
    for k in range(j):

        #Applies the oracle
        groverAlgorithm.unitary(oracle, range(nQubit))

        #Applies Grover's Algorithm

        groverAlgorithm.h(range(nQubit))
        groverAlgorithm.x(range(nQubit))
        groverAlgorithm.i(range(nQubit-1))
        #This needs to be a multicontrolled Z-Gate. For some reason, using that gate gave an error
        #So, since Z = HXH, we used a multicontrolled X gate instead.
        #This idea came from the qiskit page for Grover's Algorithm
        groverAlgorithm.h(nQubit-1)
        #Need to make this a toffoli gate that uses nQubit-1 control bits on the final qubit
        #groverAlgorithm.toffoli(0, 1, 2)
        c3z = XGate().control(3)
        groverAlgorithm.append(c3z, [0,1,2,3])
        groverAlgorithm.i(range(nQubit-1))
        groverAlgorithm.h(nQubit-1)

        groverAlgorithm.x(range(nQubit))
        groverAlgorithm.h(range(nQubit))
    #Observe the outcome of the algorithm
    groverAlgorithm.measure_all()

    #This simulates the quantum circuit. This code came from the Qiskit page for Grover's Algorithm
    sim = Aer.get_backend('qasm_simulator')  
    qobj = assemble(groverAlgorithm)  
    result = sim.run(qobj).result()  

    #Gets the probabilities from measuring the bits
    counts = result.get_counts()

    #Gets the index with the maximum probability
    indexResult = int(max(counts, key=counts.get), 2)

    #If a marked state is identified, set y equal to it, then keep looping through the algorithm
    if (L[indexResult] < L[y]):
        y = indexResult
        m = 1
    #If a marked state is not identified, redefine m, then try to find a marked state again.
    else:
        i = i - 1
        m = min(l * m, nQubit)
    #Show the histograms for probabilities
    plot_histogram(counts)
    plt.show()
# and display it on a histogram

#Print the resulting index
print("Index for minimum", y)

