# https://github.com/Dheasra/TPIV---EPFL/blob/6d1f3dfa4eb35360b6447ea81c6c067b9f37e3ac/Grover%20TPIVa/set6.py
#exercise set 6 - Quantum Computation & Quantum Info
import math
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice
import matplotlib.pyplot as plt
import matplotlib

#QFT: straight up ripped from the textbook
def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    #swaps
    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            circ.cp(-math.pi/float(2**(j-m)), m, j)
        circ.h(j)


N = 4 #nbr of qubits
Nc = 3 #nbr of classical bits
# phase = math.pi/4 #phase of the phase gate (e^i*phase)
phase = 2*pi/3  #for this phase with 4 qubits (only 3 qubits in the first register), the result is peaked around 011, which translate to 3 in decimal.
                #The phase is thus 3*1/2^(nbr of qubits in the first register) so 3/2^3 = 3/8 which is the closest value to 1/3 = 2pi/(2pi*3) 

qpe = QuantumCircuit(N,Nc) #initializing
#preparing the eigenstate of the arbitrary phase gate on qubit 3 (=the fourth one)
qpe.x(N-1)

for qubit in range(N-1): #applying H to the first 3 qubits
    qpe.h(qubit)


repet = 1
for cnt_qubit in range(N-1):
    for i in range(repet): #applies a C-U on the circuit from cnt_qubit to qubit 3 for repet times
        qpe.cp(phase, cnt_qubit, N-1);
    repet *= 2 #doubles the nbr of times the C-U will be applied next time

#visualization of the circuit
qpe.draw()

qpe.barrier()
qft_dagger(qpe, N-1) #apply inverse qft

#measurement
qpe.barrier()
for n in range(N-1):
    qpe.measure(n,n)

#visualization of the circuit
# qpe.draw() #as a picture (does not work for some reason)
print(qpe) #in the terminal

#=== Results ===
backend = Aer.get_backend('qasm_simulator') #selection of the device on which to execute the circuit
shots = 2048 #nbr of runs of the circuit
results = execute(qpe, backend = backend, shots = shots).result()
answer = results.get_counts()

#ploting
plot_histogram(answer)
plt.show()
