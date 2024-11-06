# https://github.com/thiagom123/Circuito-Quantico-Parametrizado/blob/0866c7cf39a42ff5d446b8c02f7f943563e530b0/CircuitoEncoder.py
import numpy as np
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from gates import Rz, Ry, H

def CircuitoEncoder(X, n):
    '''
    Circuito enconder que leva x_i em ( cos (x_i)
    :param X: Vetor de características
    :param n: Dimensão do Vetor X
    :return: Circuito codificado com n qubits
    '''
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    circuito1: QuantumCircuit = QuantumCircuit(qr, cr)

    # --------------------
    # Seu código aqui
    for i in range(n):
        circuito1.ry(X[i]*np.pi,qr[i])
    # --------------------

    return circuito1

if __name__ == '__main__':
    circuito = CircuitoEncoder([1], 1)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()