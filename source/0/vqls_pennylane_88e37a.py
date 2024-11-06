# https://github.com/JorgeAGR/nmsu-course-work/blob/6cd204abbc074734fb7e8ca0e693a15e1cbe4ede/PHYS520/Project/vqls_pennylane.py
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import math
import random
import numpy as np
from scipy.optimize import minimize
#%config InlineBackend.figure_format = 'svg' # Makes the images look nice

backend = Aer.get_backend('qasm_simulator')
n_qubits = 1  # Number of system qubits.
n_shots = 10000  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

c = np.array([1.0, 0.2, 0.2])

qr = QuantumRegister(tot_qubits)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)
shots = 10000

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for idx in range(n_qubits):
        qc.h(qr[idx])

def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        # Identity operation
        None

    elif idx == 1:
        qc.cz(qr[ancilla_idx], qr[1])

    elif idx == 2:
        qc.cx(qr[ancilla_idx], qr[0])
        
def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
        qc.h(qr[idx])

    # A very minimal variational circuit.
    for idx, element in enumerate(weights):
        qc.ry(element, qr[idx])
        
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    # First Hadamard gate applied to the ancillary qubit.
    qc.h(qr[ancilla_idx])

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i" phase gate.
    if part == "Im" or part == "im":
        qc.u1(-np.pi/2, qr[ancilla_idx])

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qc.cz(qr[ancilla_idx], qr[j])

    # Unitary U_b associated to the problem vector |b>.
    U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    CA(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qc.h(qr[ancilla_idx])

    # Expectation value of Z for the ancillary qubit.
    qc.measure(qr[ancilla_idx], cr[0])
    job = execute(qc, backend, shots=shots)
    results = job.result().get_counts()
    for idx in range(tot_qubits):
       qc.reset(qr[idx])
    if '1' in results.keys():
        return results['1']/shots
    else:
        return 0

def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""
    
    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag

def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    """Local version of the cost function, which tends to zero when A |x> is proportional to |b>."""
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits)