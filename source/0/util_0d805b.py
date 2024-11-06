# https://github.com/Robinbux/AI-Projects/blob/edde9e02ad21263ad21ad0d3bff64e78c556d587/QKNN/util.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator

import numpy as np

def perform_unitary_check(W):
    #array_to_latex(W, pretext="\\text{W} = ")
    W_transposed = np.transpose(W)
    #array_to_latex(W_transposed, pretext="\\text{W}^* = ")
    result = np.dot(W, W_transposed).round()
    #array_to_latex(result, pretext="W^*W = ")
    return np.all(np.equal(result, np.eye(result.shape[0])))

# Using Householder transformation https://en.wikipedia.org/wiki/Householder_transformation
def create_unitary(v):
    dim = v.size
    # Return identity if v is a multiple of e1
    if v[0][0] and not np.any(v[0][1:]):
        return np.identity(dim)
    e1 = np.zeros(dim)
    e1[0] = 1
    w = v/np.linalg.norm(v) - e1
    return np.identity(dim) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))


#----------------------------------------------------------------
# GATE CREATION
#----------------------------------------------------------------

# Variables
m = 2
n = 4
b = 4

psi = np.array([[0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0]])
phi_zero = np.array([[0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]])
phi_one = np.array([[0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0]])
phi_two = np.array([[1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]])
phi_three = np.array([[1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0]])

matrix_psi = create_unitary(psi)
matrix_phi_zero = create_unitary(phi_zero)
matrix_phi_one = create_unitary(phi_one)
matrix_phi_two = create_unitary(phi_two)
matrix_phi_three = create_unitary(phi_three)

phi_matrices = [matrix_phi_zero, matrix_phi_one, matrix_phi_two, matrix_phi_three]

def create_w_oracle(j, transpose=False):
    W_state_preparation_oracle_circuit = QuantumCircuit(m + n)
    
    matrix = phi_matrices[j].T if transpose else phi_matrices[j]
    
    matrix_operator = Operator(matrix)
    W_state_preparation_oracle_circuit.unitary(matrix_operator, list(range(m, m+n)), label='W')
    
    W_oracle = W_state_preparation_oracle_circuit.to_gate()
    W_oracle.name = "$W^\dagger$" if transpose else "$W$"
    return W_oracle

def create_v_oracle():
    V_state_preparation_oracle_circuit = QuantumCircuit(n)
    V_state_preparation_oracle_circuit.unitary(matrix_psi, list(range(0, n)), label='V')
    
    V_oracle = V_state_preparation_oracle_circuit.to_gate()
    V_oracle.name = "$V$"
    return V_oracle


def create_u_oracle(transpose=False):
    V_oracle = create_v_oracle()

    train_register = QuantumRegister(n, 'train')
    test_register = QuantumRegister(n, 'test')
    B_register = QuantumRegister(1, 'b')
    U_oracle_circuit = QuantumCircuit(train_register, test_register, B_register)
    
    U_oracle_circuit.append(V_oracle, *[test_register[:]])
    U_oracle_circuit.h(B_register[0])

    # Ist das richtig?
    for qubit in range (n):
        U_oracle_circuit.cswap(B_register[0], train_register[qubit], test_register[qubit])
    U_oracle_circuit.h(B_register[0])
    
    U_oracle = U_oracle_circuit.to_gate()
    U_oracle.name = "$U^\dagger$" if transpose else "$U$"
    return U_oracle




# Conditional Phase Shift Gate
def create_s0_gate():
    s0_circuit = QuantumCircuit(2 * n + 1)
    
    zero_ket = np.zeros((2**(2*n + 1), 1))
    zero_ket[0] = 1
    zero_bra = zero_ket.T
    zero_ket_bra = np.outer(zero_ket, zero_bra)
    
    matrix = np.identity(2**(2*n + 1)) - 2*zero_ket_bra
    
    s0_circuit.unitary(matrix, list(range(0, 2 * n + 1)), label='V')
    s0_gate = s0_circuit.to_gate()
    s0_gate.name = "$S_0$"
    return s0_gate

def create_g_gate(j):
    W_oracle = create_w_oracle(j)
    U_oracle = create_u_oracle()
    W_oracle_dagger = create_w_oracle(j, transpose = True)
    U_oracle_dagger = create_u_oracle(transpose = True)
    S0_gate = create_s0_gate()
    
    index_register = QuantumRegister(m, 'index')
    train_register = QuantumRegister(n, 'train')
    test_register = QuantumRegister(n, 'test')
    B_register = QuantumRegister(1, 'b')
    
    G_circ = QuantumCircuit(index_register, train_register, test_register, B_register)
    
    G_circ.z(B_register[0])
    
    # W and U Dagger
    G_circ.append(U_oracle_dagger, [*train_register[:],*test_register[:],B_register[0]])
    G_circ.append(W_oracle_dagger, [*index_register[:],*train_register[:]])

    # S0 Conditional Phase Shift Gate
    G_circ.append(S0_gate, [*train_register[:],*test_register[:],B_register[0]])

    # W and U
    G_circ.append(W_oracle, [*index_register[:],*train_register[:]])
    G_circ.append(U_oracle, [*train_register[:],*test_register[:],B_register[0]])
    
    G_gate = G_circ.to_gate()
    G_gate.name = "$G$"
    return G_gate

def phase_estimation_on_g(j):
    G_gate_controll = create_g_gate(j).control(1)
    
    index_register = QuantumRegister(m, 'index')
    train_register = QuantumRegister(n, 'train')
    test_register = QuantumRegister(n, 'test')
    B_register = QuantumRegister(1, 'b')
    phase_register = QuantumRegister(b, 'phase')
    
    phase_est_circ = QuantumCircuit(index_register, train_register, test_register, B_register, phase_register)
    
    for phase_index in range(b):
        for gate_index in range(2**b):
            phase_est_circ.append(G_gate_controll, [*index_register[:],*train_register[:], *test_register[:], B_register[0], phase_register[phase_index]])

    phase_est_gate = phase_est_circ.to_gate()
    phase_est_gate.name = "$PhaseEstimation on G$"
    return phase_est_gate