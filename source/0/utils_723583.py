# https://github.com/BenediktRiegel/quantum-no-free-lunch/blob/059253d5aa7c3f6dcd66fef7d8562d38e6452f97/utils.py
import pennylane as qml
import numpy as np
import torch

def int_to_bin(num, num_bits):
    """
    Convert integer to binary with padding
    (e.g. (num=7, num_bits = 5) -> 00111)
    """
    b = bin(num)[2:]
    return [0 for _ in range(num_bits - len(b))] + [int(el) for el in b]

def one_hot_encoding(num, num_bits):
    """
    Returns one-hot encoding of a number
    (e.g. (num=4, num_bits=7) -> 0000100)
    """
    result = [0]*num_bits
    result[num] = 1
    return result

def normalize(point):
    return point / np.linalg.norm(point)

def tensor_product(state1: np.ndarray, state2: np.ndarray):
    result = np.zeros(len(state1)*len(state2), dtype=np.complex128)
    for i in range(len(state1)):
        result[i*len(state2):i*len(state2)+len(state2)] = state1[i] * state2
    return result

def torch_tensor_product(matrix1: torch.Tensor, matrix2: torch.Tensor, device='cpu'):
    result = torch.zeros((matrix1.shape[0]*matrix2.shape[0], matrix1.shape[1]*matrix2.shape[1]), dtype=torch.complex128, device=device)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i*matrix2.shape[0]:i*matrix2.shape[0]+matrix2.shape[0], j*matrix2.shape[1]:j*matrix2.shape[1]+matrix2.shape[1]] = matrix1[i, j] * matrix2
    return result

def adjoint_unitary_circuit(unitary):
    """
    Generates a circuit corresponding to the adjoint of the input matrix
    """
    from qiskit import QuantumCircuit, Aer, transpile

    unitary = np.conj(np.array(unitary)).T

    qbits = int(np.log2(len(unitary)))
    sv_backend = Aer.get_backend('statevector_simulator')

    qc = QuantumCircuit(qbits)
    qc.unitary(unitary, range(qbits))
    qc_transpiled = transpile(qc, backend=sv_backend, basis_gates=sv_backend.configuration().basis_gates,
                              optimization_level=3)
    return qml.from_qiskit(qc_transpiled)

