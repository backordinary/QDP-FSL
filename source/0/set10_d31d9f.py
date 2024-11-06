# https://github.com/Dheasra/TPIV---EPFL/blob/6d1f3dfa4eb35360b6447ea81c6c067b9f37e3ac/Grover%20TPIVa/set10.py
#exercise set 6 - Quantum Computation & Quantum Info
import math
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer import QasmSimulator, noise
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
# from qiskit.providers.aer import noise
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice
import matplotlib.pyplot as plt
import matplotlib

N = 3 #nbr of qubits
Na = N-1 #nbr of ancilla qubits
Nc = 3 #nbr of classical bits
# phase = math.pi/4 #phase of the phase gate (e^i*phase)
# phase = 2*pi/3  #for this phase with 4 qubits (only 3 qubits in the first register), the result is peaked around 011, which translate to 3 in decimal.
                #The phase is thus 3*1/2^(nbr of qubits in the first register) so 3/2^3 = 3/8 which is the closest value to 1/3 = 2pi/(2pi*3)

qr = QuantumRegister(N, 'code')
ar = QuantumRegister(Na, 'ancilla')
cr = ClassicalRegister(Nc)
qecc = QuantumCircuit(qr, ar, cr)

# initializing the first register into a repition code
qecc.h(0)
qecc.cx(0,1)
qecc.cx(0,2)

qecc.barrier()

# error detection
#my version
qecc.ccx(0,1,3)
qecc.ccx(1,2,4)

# #Correct version
# qecc.cx(0,3)
# qecc.cx(1,3)
# qecc.cx(1,4)
# qecc.cx(2,4)

qecc.barrier()

# error correction
qecc.x(3)
qecc.ccx(3,4,0)
qecc.x(4)
qecc.ccx(3,4,1)
qecc.x(3)
qecc.ccx(3,4,2)
qecc.x(4)

#measurement
qecc.barrier()
for n in range(N-1):
    qecc.measure(n,n)

#=== Results ===
backend = Aer.get_backend('qasm_simulator') #selection of the device on which to execute the circuit
shots = 2048 #nbr of runs of the circuit
results = execute(qecc, backend = backend, shots = shots).result()
answer = results.get_counts()

#ploting
plot_histogram(answer)
plt.show()
