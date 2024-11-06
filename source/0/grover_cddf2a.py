# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw5/grover.py
import qiskit
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def phase_oracle(qc, n, indices_to_mark):
    
    oracle_matrix = np.identity(2**n)
    for i in indices_to_mark:
        oracle_matrix[i,i] = -1 # adds negative phase to solution state, shrinks other states amplitude by lowering average
        
    qc.unitary(qiskit.quantum_info.Operator(oracle_matrix), range(n))

# first step in circuit - put all inputs in superposition
def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits): # transform s back to 0
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
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "$U_s$"
    return U_s


if __name__ == '__main__':
    n = 4
    #inputs = QuantumRegister(1)  # input
    #outputs = QuantumRegister(2) # output
    #c = ClassicalRegister(3)
    grover_circuit = QuantumCircuit(n)

    # this state (s vector) and winner state (w vector) span a 2D space, the vector w - s = s' is perpendicular to w
    # the angle s makes with s' is theta, s = sin(theta)w + cos(theta)s', theta = arcsin(s|w) = arcsin(1/sqrt(N))
    grover_circuit = initialize_s(grover_circuit, range(n))
    
    # the oracle U_f reflects s about the s' axis
    phase_oracle(grover_circuit, n, [13])

    # diffusion U_s = 2s*s - I, reflects U_f*s about s vector, difference of 2*theta between them
    # U_sU_f*s is now 3*theta above s' axis, these reflections are updating the amplitude
    # when its reflected below s' it has negative amplitude, then its positive amplitude increases from original
    # after second relection, 3*theta means 3 times original amplitude, this is one timestep t, theta_t = (2t + 1)theta
    # since this is reflection about s, add negative phase to every state orthogonal to s
    grover_circuit.append(diffuser(n), range(n))
    grover_circuit.measure_all()
    
    backend = Aer.get_backend('qasm_simulator')
    results = execute(grover_circuit, backend=backend, shots=1024).result()
    answer = results.get_counts()
    plot_histogram(answer)
    plt.show()