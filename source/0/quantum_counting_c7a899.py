# https://github.com/MarvOdo/Quantum-Counting/blob/ffbb4340042c0a2aa2ff05849a66d0c9114924f3/quantum-counting.py
#Implementation of Quantum Counting Algorithm in Qiskit
#By Marvin Odobashi

import numpy as np
from qiskit import *
from qiskit.circuit.library.standard_gates import ZGate

#qubits that determine precision of algorithm
p = 7
#qubits to represent possible inputs to Boolean Function
n = 4
upper = QuantumRegister(p, 'upper')
#1 ancilla qubit to phase shift "correct" states with oracle
lower = QuantumRegister(n+1, 'lower')
measurements = ClassicalRegister(p, 'measurements')
qc = QuantumCircuit(upper, lower, measurements)

#There exists some unknown n-bit input Boolean Function F: {0, 1}^n -> {0, 1}
#Assume we have this oracle that will phase shift solution states
#In this speific case I have picked n=4, but the "solution state selectors" have been defined for a general n
def oracle(qc):
    w = qc.width()
    
    #controlled z gate (phase shift state if it is a solution)
    ctrlz = ZGate().control(w-1)
    
    #ctrlz for |1111..>
    qc.append(ctrlz, range(w))
    
    #ctrlz for |0000..>
    qc.x(range(w-1))
    qc.append(ctrlz, range(w))
    qc.x(range(w-1))
    
    #ctrlz for |10..01>
    qc.x(range(1, w-2))
    qc.append(ctrlz, range(w))
    qc.x(range(1, w-2))
    
    #ctrlz for |01..10>
    qc.x([0, w-2])
    qc.append(ctrlz, range(w))
    qc.x([0, w-2])
    
    #ctrlz for |0111..>
    #qc.x([0])
    #qc.append(ctrlz, range(w))
    #qc.x([0])
    
    #ctrlz for |1011..>
    #qc.x([1])
    #qc.append(ctrlz, range(w))
    #qc.x([1])
    
    return qc

#Grover operator to be repeated
def grover(qc):
    w = qc.width()
    
    #oracle
    qc = oracle(qc)
    
    #Hadamard-All
    qc.h(range(w-1))
    
    #Check if all are 0
    ctrlz = ZGate().control(w-1)
    qc.x(range(w-1))
    qc.append(ctrlz, range(w))
    qc.x(range(w-1))
    
    #Hadamard-All
    qc.h(range(w-1))
    
    return qc

from qiskit.circuit.library import QFT

#quantum counting
def quantumCount(qc):
    #create controlled gate version of grover operation
    ctrlGrover = grover(QuantumCircuit(lower)).to_gate(label='CG').control(1)
    
    #initialize circuit, flip last (ancilla) qubit so it can be used for phase shifting by oracle
    qc.h(upper)
    qc.h(lower[0:-1])
    qc.x(lower[-1])
    
    #repeat controlled grover 2^i times for i in {0, p-1}
    for i in range(p):
        for j in range(2**i):
            #controlled grover, ith upper qubit as control, lower register is operated on
            qc.append(ctrlGrover, [qc.qubits[i]] + qc.qubits[p:])
    
    #apply inverse QFT on upper register
    #using built-in QFT for convenience
    iqft = QFT(num_qubits=p, inverse=True).to_gate(label='iQFT')
    qc.append(iqft, qc.qubits[:p])
    
    #measure upper register
    qc.measure(range(p), range(p))
    
    return qc

#full circuit
final = quantumCount(qc)
#final.draw('mpl')

from qiskit import transpile
from qiskit.providers.aer import AerSimulator

backend = AerSimulator()
qc_compiled = transpile(final, backend)
job_sim = backend.run(qc_compiled, shots=1024)
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)

#from qiskit.visualization import plot_histogram
#plot_histogram(counts)

#take highest counted state
measured_n = int(max(counts, key=counts.get), 2)
#get theta
theta = 2*np.pi*measured_n/(2**p)
#compute number of solutions
num_sol = (2**n)*np.cos(theta/2)**2
#compute error (formula from https://qiskit.org/textbook/ch-algorithms/quantum-counting.html#finding_m)
N = 2**n
M = N * np.sin(theta / 2)**2
error = (np.sqrt(2*M*N) + N/(2**(p)))*(2**(-p+1))
print(f"""Actual Number of Solutions = 4\n
Quantum Counting Number of Solutions = {num_sol:.1f}\n
Quantum Counting Error < {error:.2f}""")




