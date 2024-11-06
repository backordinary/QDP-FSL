# https://github.com/jlgingrich/quantum-grover/blob/9fd05141e06332e40179ebb099dc9b63d0bab3ce/sudoku_quantum_counting.py
# import Qiskit
from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.visualization import plot_histogram, plot_state_city

# import helpers 
import numpy as np
from numpy import pi, sin, sqrt

clause_list = [[0,1],
               [0,2],
               [1,3],
               [2,3]]

def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)
    
def sudoku_oracle():
    var_qubits = QuantumRegister(4, name='v')
    clause_qubits = QuantumRegister(4, name='c')
    output_qubit = QuantumRegister(1, name='out')
    qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit)
    # Compute clauses
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1
    # Flip 'output' bit if all clauses are satisfied
    qc.mct(clause_qubits, output_qubit)
    # Uncompute clauses to reset clause-checking bits to 0
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1
    # print("Oracle:\n", qc.draw()) # Uncomment to peek at oracle
    U_f = qc.to_gate()
    U_f.name = "$U_f$"
    return U_f

# define diffuser gate
def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # print("Diffuser:\n", qc.draw()) # Uncomment to peek at diffuser
    # Return as gate
    U_s = qc.to_gate()
    U_s.name = "$U_s$"
    return U_s

# define Grover iteration circuit
def grover_iteration():
    # Do circuit
    qc = QuantumCircuit(9)
    # append Oracle
    qc.append(sudoku_oracle(), [0,1,2,3,4,5,6,7,8])
    # append Diffuser
    qc.append(diffuser(4), [0,1,2,3])
    # print("Grover Iteration:\n", qc.draw()) # Uncomment to peek
    grit = qc.to_gate()
    grit.name = " Grover "
    return grit

# Create controlled-Grover gate
cgrit = grover_iteration().control()
cgrit.label = "Control"

# define quantum fourier transform circuit
def qft(n):
    """Creates an n-qubit QFT circuit"""
    circuit = QuantumCircuit(n)
    def swap_registers(circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    def qft_rotations(circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cu1(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

# define inverse quantum fourier transform for 4 qubits
qft_dagger = qft(4).to_gate().inverse()
qft_dagger.label = "QFT†"

# define solution counter
def calculate_M(measured_int, t, n):
    """For Processing Output of Quantum Counting"""
    # Calculate Theta
    theta = (measured_int/(2**t))*pi*2
    # Calculate No. of Solutions
    N = 2**n
    M = N * (sin(theta/2)**2)
    # Calculate Upper Error Bound
    m = t - 1 #Will be less than this (out of scope) 
    err = (sqrt(2*M*N) + N/(2**(m-1)))*(2**(-m))
    print("Theta = %.5f" % theta)
    print("No. of Solutions = %.1f" % (N-M))
    print("Error < %.2f" % err)

# ------------------------------------------------------
# ----------------------- SCRIPT -----------------------
# ------------------------------------------------------

t = 4   # no. of counting qubits
n = 9   # no. of searching qubits
x = 5   # number of non-oracle qubits required to define oracle
qc = QuantumCircuit(n+t, t) # Circuit with n+t qubits and t classical bits

# Initialize last qubit in state |->
qc.initialize([1, -1]/np.sqrt(2), 12)

# Initialise all qubits to |+>
for qubit in range(t+n-x):
    qc.h(qubit)

# Begin controlled Grover iterations
iterations = 1
for qubit in range(t):
    for i in range(iterations):
        qc.append(cgrit, [qubit] + [*range(t, n+t)])
    iterations *= 2
# Do inverse QFT on counting qubits
qc.append(qft_dagger, range(t))

# Measure counting qubits
qc.measure(range(t), range(t))

qc.draw(output='mpl', filename='sudoku_quantum_counting.png')

# Execute and see results
emulator = Aer.get_backend('qasm_simulator')
job = execute(qc, emulator, shots=2048 )
hist = job.result().get_counts()

# Process to expected value
measured_str = max(hist, key=hist.get)
measured_int = int(measured_str,2)
print("Register Output = %i" % measured_int)
calculate_M(measured_int, t, n-x)