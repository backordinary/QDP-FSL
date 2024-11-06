# https://github.com/varsill/totem_simulator/blob/2f82a57d87f299c5abef1e1ad788db65268e60a0/common.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, assemble
from qiskit.quantum_info import Statevector
import math
import numpy as np
from qiskit.algorithms.linear_solvers.hhl import HHL


def inner_prod(vec1, vec2):
    # first check if lengths are equal
    if len(vec1) != len(vec2):
        raise ValueError('Lengths of states are not equal')
    N = len(vec1)
    nqubits = math.ceil(np.log2(N))

    # normalizing
    vec1norm = np.linalg.norm(vec1)
    vec2norm = np.linalg.norm(vec2)
    vec1 = vec1 / vec1norm
    vec2 = vec2 / vec2norm

    circ = QuantumCircuit(nqubits + 1)
    vec = np.concatenate((vec1, vec2)) / np.sqrt(2)

    circ.initialize(vec, range(nqubits + 1))
    circ.h(nqubits)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend, backend_options={"zero_threshold": 1e-20})

    result = job.result()
    o = np.real(result.get_statevector(circ))

    m_sum = 0
    for l in range(N):
        m_sum += o[l]**2

    return (2 * m_sum - 1) * vec1norm * vec2norm


def multiply(A, B):
    result = np.zeros(shape=(A.shape[0], B.shape[1]))
    for i, row in enumerate(A):
        for j, col in enumerate(B.T):
            result[i][j] = inner_prod(row, col)

    return result


def solve_linear_equation(X, b):
    norm = np.linalg.norm(b)
    b = b / norm
    original_dimension = X.shape
    X, b, truncate_powerdim, truncate_hermitian = make_symetric(X, b)
    # X = X.astype(np.float64)
    # b = b.astype(np.float64)
    naive_hhl_solution = HHL().solve(X, b)
    naive_sv = Statevector(naive_hhl_solution.state).data
    vec = np.real(naive_sv)
    if truncate_hermitian:
        half_dim = int(vec.shape[0] / 2)
        vec = vec[:half_dim]
        if truncate_powerdim:
            vec = vec[original_dimension]
    return (norm * naive_hhl_solution.euclidean_norm *
            vec / np.linalg.norm(vec))[:original_dimension[1]]
    

def make_symetric(matrix: np.ndarray,
                    vector: np.ndarray):
    """Resizes matrix if necessary
    Args:
        matrix: the input matrix of linear system of equations
        vector: the input vector of linear system of equations
    Returns:
        new matrix, vector, truncate_powerdim, truncate_hermitian
    Raises:
        ValueError: invalid input
    """

    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)

    if matrix.shape[0]!=matrix.shape[1]:
        to_add = (np.random.rand(matrix.shape[0], matrix.shape[0]-matrix.shape[1])-0.5)*0.01
        matrix = np.concatenate((matrix, to_add), axis=1)
    if matrix.shape[0] != len(vector):
        raise ValueError("Input vector dimension does not match input "
                            "matrix dimension!")

    truncate_powerdim = False
    truncate_hermitian = False
    orig_size = None
    if orig_size is None:
        orig_size = len(vector)

    is_powerdim = np.log2(matrix.shape[0]) % 1 == 0
    if not is_powerdim:
        matrix, vector = expand_to_powerdim(matrix, vector)
        truncate_powerdim = True

    is_hermitian = np.allclose(matrix, matrix.conj().T)
    if not is_hermitian:
        matrix, vector = expand_to_hermitian(matrix, vector)
        truncate_hermitian = True

    return matrix, vector, truncate_powerdim, truncate_hermitian

def expand_to_powerdim(matrix: np.ndarray, vector: np.ndarray):
    """ Expand a matrix to the next-larger 2**n dimensional matrix with
    ones on the diagonal and zeros on the off-diagonal and expand the
    vector with zeros accordingly.
    Args:
        matrix: the input matrix
        vector: the input vector
    Returns:
        the expanded matrix, the expanded vector
    """
    mat_dim = matrix.shape[0]
    next_higher = int(np.ceil(np.log2(mat_dim)))
    new_matrix = np.identity(2 ** next_higher)
    new_matrix = np.array(new_matrix, dtype=complex)
    new_matrix[:mat_dim, :mat_dim] = matrix[:, :]
    matrix = new_matrix
    new_vector = np.zeros((1, 2 ** next_higher))
    new_vector[0, :vector.shape[0]] = vector
    vector = new_vector.reshape(np.shape(new_vector)[1])
    return matrix, vector

def expand_to_hermitian(matrix: np.ndarray,
                        vector: np.ndarray):
    """ Expand a non-hermitian matrix A to a hermitian matrix by
    [[0, A.H], [A, 0]] and expand vector b to [b.conj, b].
    Args:
        matrix: the input matrix
        vector: the input vector
    Returns:
        the expanded matrix, the expanded vector
    """
    #
    half_dim = matrix.shape[0]
    full_dim = 2 * half_dim
    new_matrix = np.zeros([full_dim, full_dim])
    new_matrix = np.array(new_matrix, dtype=complex)
    new_matrix[0:half_dim, half_dim:full_dim] = matrix[:, :]
    new_matrix[half_dim:full_dim, 0:half_dim] = matrix.conj().T[:, :]
    matrix = new_matrix
    new_vector = np.zeros((1, full_dim))
    new_vector = np.array(new_vector, dtype=complex)
    new_vector[0, :vector.shape[0]] = vector.conj()
    new_vector[0, vector.shape[0]:] = vector
    vector = new_vector.reshape(np.shape(new_vector)[1])
    return matrix, vector